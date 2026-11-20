"""Model-independent discovery of optional registered training extensions."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from functools import wraps
from collections.abc import Mapping

from ovlab_policy_sdk.training import CheckpointPreparationPlugin, TrainingPlugin

from .policy_plugins import registered_plugins
from .training_errors import CheckpointBundleError, TrainingProfileError, TrainingRuntimeError


def _extensions(repository_root=None):
    seen = set()
    for owner in registered_plugins(repository_root):
        declaration = owner.descriptor.get("training")
        if not isinstance(declaration, dict):
            continue
        key = (declaration.get("entrypoint"), tuple(declaration.get("implementations", ())))
        if key not in seen:
            seen.add(key)
            yield owner, declaration


def _load(owner, entrypoint, contract):
    candidates = owner.expose_sources()
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise TrainingRuntimeError("registered extension requires module:object entrypoint")
    module, symbol = entrypoint.split(":", 1)
    try:
        imported = importlib.import_module(module)
        owner.verify_loaded_module(imported, entrypoint, candidates=candidates)
        implementation = getattr(imported, symbol)
        if isinstance(implementation, type):
            implementation = implementation()
    except (ImportError, AttributeError, TypeError) as exc:
        raise TrainingRuntimeError(f"cannot load registered extension {entrypoint}: {exc}") from exc
    if not isinstance(implementation, contract) or implementation.api_version != "1":
        raise TrainingRuntimeError(f"registered extension {entrypoint} must implement {contract.__name__} API 1")
    return _BoundExtension(implementation)


def training_registration(implementation, repository_root=None):
    matches = [(owner, declaration) for owner, declaration in _extensions(repository_root)
               if implementation in declaration.get("implementations", ())]
    if len(matches) != 1:
        raise TrainingProfileError(f"trainer implementation {implementation!r} must have exactly one registered provider")
    return matches[0]


def backend_for_profile(document, repository_root=None):
    trainer = document.get("trainer") if isinstance(document, Mapping) else None
    if not isinstance(trainer, Mapping) or not isinstance(trainer.get("implementation"), str):
        raise TrainingProfileError("training profile requires trainer.implementation")
    owner, declaration = training_registration(trainer["implementation"], repository_root)
    return _load(owner, declaration.get("entrypoint"), TrainingPlugin)


def backend_for_plan(plan, repository_root=None):
    scientific = plan.get("scientific", {})
    profile = scientific.get("profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("trainer"), Mapping):
        return backend_for_profile(profile, repository_root)
    trainer = scientific.get("trainer", {})
    if isinstance(trainer, Mapping) and isinstance(trainer.get("id"), str):
        owner, declaration = training_registration(trainer["id"], repository_root)
        return _load(owner, declaration.get("entrypoint"), TrainingPlugin)
    # Legacy immutable plans predate trainer metadata. A single installed
    # provider can explicitly recognize them; ambiguous projects are rejected.
    matches = list(_extensions(repository_root))
    if len(matches) != 1:
        raise CheckpointBundleError("legacy training plan lacks a unique registered trainer identity")
    owner, declaration = matches[0]
    return _load(owner, declaration.get("entrypoint"), TrainingPlugin)


def checkpoint_registration(checkpoint, repository_root=None):
    loader = checkpoint.get("expected_loader")
    matches = [(owner, declaration) for owner, declaration in _extensions(repository_root)
               if loader in declaration.get("expected_loaders", ())]
    if len(matches) != 1:
        raise CheckpointBundleError(f"checkpoint loader {loader!r} must have exactly one registered provider")
    return matches[0]


def backend_for_checkpoint(checkpoint, repository_root=None):
    owner, declaration = checkpoint_registration(checkpoint, repository_root)
    return _load(owner, declaration.get("entrypoint"), TrainingPlugin)


def preparation_for_checkpoint(checkpoint, repository_root=None):
    owner, _ = checkpoint_registration(checkpoint, repository_root)
    declaration = owner.descriptor.get("preparation") or {}
    if checkpoint.get("expected_loader") not in declaration.get("expected_loaders", ()):
        raise CheckpointBundleError(f"plugin has no checkpoint preparation for loader {checkpoint.get('expected_loader')!r}")
    return _load(owner, declaration.get("entrypoint"), CheckpointPreparationPlugin), declaration


def training_image(document, repository_root=None):
    implementation = document["trainer"]["implementation"]
    _, declaration = training_registration(implementation, repository_root)
    image = declaration.get("images", {}).get(implementation)
    if not isinstance(image, dict) or any(not isinstance(image.get(key), str) for key in ("role", "image", "image_env")):
        raise TrainingProfileError(f"trainer {implementation!r} has no declared training image")
    return image


def extension_symbol(namespace, name, repository_root=None):
    """Resolve old helper imports through declared module exports, without model names."""
    if name.startswith("__"):
        raise AttributeError(name)
    modules = {}
    for owner, declaration in _extensions(repository_root):
        module_name = declaration.get("modules", {}).get(namespace)
        if module_name:
            modules[module_name] = owner
    if len(modules) != 1:
        raise AttributeError(f"registered project helper {namespace}.{name} has no unique provider")
    module_name, owner = next(iter(modules.items()))
    owner.expose_sources()
    try:
        return getattr(importlib.import_module(module_name), name)
    except AttributeError as exc:
        raise AttributeError(name) from exc


class _BoundExtension:
    """Supply SDK callbacks only while a registered extension hook executes."""
    def __init__(self, implementation):
        self.implementation = implementation

    def __getattr__(self, name):
        value = getattr(self.implementation, name)
        if not callable(value):
            return value
        @wraps(value)
        def call(*args, **kwargs):
            from ovlab_policy_sdk.training_services import training_host_scope
            from ovlab_policy_sdk.plugin import (PolicyPluginCompatibilityError,
                                                 PolicyPluginReferenceError, PolicyPluginValidationError)
            from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
            with training_host_scope(_host_services()):
                try:
                    return value(*args, **kwargs)
                except PolicyPluginCompatibilityError as exc:
                    raise ConfigCompatibilityError(str(exc)) from exc
                except PolicyPluginReferenceError as exc:
                    raise ConfigReferenceError(str(exc)) from exc
                except PolicyPluginValidationError as exc:
                    raise ConfigSchemaError(str(exc)) from exc
        return call


def _host_services():
    from . import checkpointing
    from .schema import validate
    from .training_runs import TrainingRunStore, CheckpointBundleStore
    from .run_references import resolve_run_reference_tree
    def image_environment(repository_root, environment, env_file, variable):
        from .deployment import ComposeDeployment
        return ComposeDeployment(repository_root, environment=environment)._environment_value(
            SimpleNamespace(env_file=env_file), variable)
    return SimpleNamespace(
        checkpoint_spec=checkpointing.CheckpointSpec,
        checkpoint_spec_from_snapshot=checkpointing.checkpoint_spec_from_snapshot,
        default_global_cache=checkpointing.default_global_cache,
        resolve_finalized_training_checkpoint=checkpointing.resolve_finalized_training_checkpoint,
        verify_checkpoint=checkpointing.verify_checkpoint,
        local_checkpoint_override=checkpointing.local_checkpoint_override,
        training_store=TrainingRunStore, checkpoint_store=CheckpointBundleStore,
        resolve_run_reference=resolve_run_reference_tree, validate_document=validate,
        image_environment=image_environment,
    )


def finalizer_image(document, repository_root=None):
    implementation = document["trainer"]["implementation"]
    _, declaration = training_registration(implementation, repository_root)
    image = declaration.get("finalizer")
    if not isinstance(image, dict) or any(not isinstance(image.get(key), str) for key in ("role", "image", "image_env")):
        raise TrainingProfileError(f"trainer {implementation!r} has no declared checkpoint finalizer image")
    return image
