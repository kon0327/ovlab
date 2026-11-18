"""Versioned policy discovery. Reading a registry never imports model code."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError

DEFAULT_REGISTRY = Path(__file__).with_name("plugins") / "registry.json"
API_VERSION = "1"
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_ROOT = ContextVar("ovlab_plugin_repository_root", default=None)
_UNSET = object()
_REGISTRY = ContextVar("ovlab_plugin_registry_override", default=_UNSET)


@contextmanager
def registry_scope(repository_root, environment=None):
    token = _ROOT.set(repository_root or _ROOT.get())
    override = _REGISTRY.set(environment.get("OVLAB_PLUGIN_REGISTRY")) if environment is not None else None
    try:
        yield
    finally:
        _ROOT.reset(token)
        if override is not None:
            _REGISTRY.reset(override)


def with_registry(method):
    """Apply an orchestrator's explicit environment without mutating os.environ."""
    @wraps(method)
    def scoped(self, *args, **kwargs):
        values = kwargs.get("environment", getattr(self, "environment", None))
        with registry_scope(self.repository_root, values):
            return method(self, *args, **kwargs)
    return scoped


def _root(repository_root=None):
    return Path(repository_root or _ROOT.get() or os.environ.get("OVLAB_ROOT", Path.cwd())).resolve()


def registry_path(repository_root=None):
    root = _root(repository_root)
    selected = _REGISTRY.get()
    configured = os.environ.get("OVLAB_PLUGIN_REGISTRY") if selected is _UNSET else selected
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else root / path
    local = root / "configs/plugins/registry.json"
    return local if local.is_file() else DEFAULT_REGISTRY


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigSchemaError(f"duplicate plugin registry key: {key}")
        result[key] = value
    return result


def registry_document(repository_root=None):
    path = registry_path(repository_root)
    try:
        document = json.loads(path.read_text(), object_pairs_hook=_unique)
    except (OSError, ValueError) as exc:
        raise ConfigSchemaError(f"cannot read plugin registry {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "ovlab.plugins/v1":
        raise ConfigSchemaError("plugin registry schema_version must be ovlab.plugins/v1")
    for name in ("sources", "plugins", "builds"):
        if not isinstance(document.get(name), dict):
            raise ConfigSchemaError(f"plugin registry {name} must be a mapping")
        if any(not _NAME.fullmatch(key) for key in document[name]):
            raise ConfigSchemaError(f"invalid identifier in plugin registry {name}")
    for name, source in document["sources"].items():
        if not isinstance(source, dict) or source.get("kind") not in {"git", "local"}:
            raise ConfigSchemaError(f"source {name} requires kind git or local")
        if source["kind"] == "git":
            if not isinstance(source.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
                raise ConfigSchemaError(f"source {name} requires a full lowercase Git commit")
            if not isinstance(source.get("url"), str) or not source["url"] or source["url"].startswith("-"):
                raise ConfigSchemaError(f"source {name} requires a Git URL")
            if source.get("path", f"external/{name}") != f"external/{name}":
                raise ConfigSchemaError(f"source {name} must use external/{name}")
        elif not isinstance(source.get("path"), str) or not source["path"]:
            raise ConfigSchemaError(f"local source {name} requires a filesystem path")
    for role, build in document["builds"].items():
        if not isinstance(build, dict) or not all(isinstance(build.get(k), str) and build[k] for k in ("dockerfile", "image")):
            raise ConfigSchemaError(f"build {role} requires dockerfile and image")
        for key in ("sources", "locks"):
            if not isinstance(build.get(key, []), list) or any(not isinstance(v, str) for v in build.get(key, [])):
                raise ConfigSchemaError(f"build {role}.{key} must be a list of strings")
        if set(build.get("sources", [])) - document["sources"].keys():
            raise ConfigSchemaError(f"build {role} references unknown sources")
        for relative in (build["dockerfile"], *build.get("locks", [])):
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ConfigSchemaError(f"build {role} paths must stay below the repository")
        if build["image"].startswith("-") or any(c.isspace() for c in build["image"]):
            raise ConfigSchemaError(f"build {role} requires a Docker image reference")
    ids = set()
    document_owners = {}
    profiles = {}
    for config_type, plugin in document["plugins"].items():
        if not isinstance(plugin, dict) or plugin.get("config_type", config_type) != config_type:
            raise ConfigSchemaError(f"invalid plugin descriptor {config_type}")
        plugin["config_type"] = config_type
        if not all(isinstance(plugin.get(k), str) and plugin[k] for k in ("id", "family", "runtime_status")):
            raise ConfigSchemaError(f"plugin {config_type} requires id, family and runtime_status")
        if plugin["runtime_status"] not in {"implemented", "experimental", "unavailable"}:
            raise ConfigSchemaError(f"plugin {config_type} has an invalid runtime_status")
        if plugin["id"] in ids:
            raise ConfigSchemaError(f"duplicate plugin ID {plugin['id']}")
        ids.add(plugin["id"])
        entrypoint = plugin.get("entrypoint")
        if plugin["runtime_status"] != "unavailable" and (not isinstance(entrypoint, str) or not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_][\w]*", entrypoint)):
            raise ConfigSchemaError(f"plugin {config_type} requires module:object entrypoint")
        if not isinstance(plugin.get("sources", []), list) or any(not isinstance(v, str) for v in plugin.get("sources", [])) or set(plugin.get("sources", [])) - document["sources"].keys():
            raise ConfigSchemaError(f"plugin {config_type} references unknown sources")
        for key in ("document_kinds", "artifact_forms"):
            if not isinstance(plugin.get(key, []), list) or any(not isinstance(v, str) or not v for v in plugin.get(key, [])):
                raise ConfigSchemaError(f"plugin {config_type}.{key} must be a list of strings")
        for kind in plugin.get("document_kinds", []):
            if kind in document_owners:
                raise ConfigSchemaError(f"document kind {kind!r} is registered by multiple plugins")
            document_owners[kind] = config_type
        deploy = plugin.get("deployment")
        if deploy is not None:
            required = ("profile", "role", "image", "image_env", "policy_service", "benchmark_service", "observation_profile")
            if not isinstance(deploy, dict) or any(not isinstance(deploy.get(k), str) or not deploy[k] for k in required):
                raise ConfigSchemaError(f"plugin {config_type} has invalid deployment metadata")
            if any(not _NAME.fullmatch(deploy[k]) for k in ("profile", "role", "policy_service", "benchmark_service")):
                raise ConfigSchemaError(f"plugin {config_type} has invalid service/profile identifier")
            if deploy["role"] not in document["builds"]:
                raise ConfigSchemaError(f"plugin {config_type} references unknown build role")
            build = document["builds"][deploy["role"]]
            if deploy["image"] != build["image"]:
                raise ConfigSchemaError(f"plugin {config_type} image differs from its build recipe")
            if set(plugin.get("sources", [])) - set(build.get("sources", [])):
                raise ConfigSchemaError(f"plugin {config_type} build omits required plugin sources")
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", deploy["image_env"]):
                raise ConfigSchemaError(f"plugin {config_type} requires a valid image environment variable")
            if deploy["policy_service"] == deploy["benchmark_service"]:
                raise ConfigSchemaError(f"plugin {config_type} policy and benchmark services must differ")
            reserved_services = {"reporting", "export", "dataset-acquire", "training-openvla", "checkpoint-finalizer"}
            if {deploy["policy_service"], deploy["benchmark_service"]} & reserved_services:
                raise ConfigSchemaError(f"plugin {config_type} uses a reserved infrastructure service name")
            if deploy["profile"] in {"reporting", "export", "dataset-acquire", "training", "training-finalize"}:
                raise ConfigSchemaError(f"plugin {config_type} uses a reserved infrastructure profile")
            if deploy["image_env"] in {"OVLAB_BENCHMARK_IMAGE", "OVLAB_REPORTING_IMAGE"}:
                raise ConfigSchemaError(f"plugin {config_type} uses a reserved infrastructure image variable")
            if type(deploy.get("gpu", True)) is not bool:
                raise ConfigSchemaError(f"plugin {config_type}.deployment.gpu must be boolean")
            if deploy["profile"] in profiles and profiles[deploy["profile"]] != deploy:
                raise ConfigSchemaError(f"deployment profile {deploy['profile']!r} has conflicting definitions")
            profiles[deploy["profile"]] = deploy
            quantization = deploy.get("quantization", ["none"])
            if not isinstance(quantization, list) or not quantization or any(v not in ("none", "4bit", "8bit") for v in quantization):
                raise ConfigSchemaError(f"plugin {config_type} has invalid quantization options")
    return document


def registry_identity(repository_root=None):
    document = registry_document(repository_root)
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class LoadedPolicyPlugin:
    descriptor: dict
    repository_root: Path
    registry: dict

    @cached_property
    def implementation(self):
        entrypoint = self.descriptor.get("entrypoint")
        if not entrypoint:
            raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} is unavailable")
        # Sources contain Python packages without changing the host environment.
        # Installed packages remain the normal path inside isolated images.
        candidates = []
        for name in self.descriptor.get("sources", []):
            source = self.registry["sources"][name]
            path = self.repository_root / "external" / name
            if source["kind"] == "local":
                original = Path(source["path"]).expanduser()
                if not original.is_absolute(): original = self.repository_root / original
                if original.is_dir(): path = original
            for candidate in (path, path / "src"):
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                candidates.append(candidate)
        module, attribute = entrypoint.split(":")
        try:
            loaded = importlib.import_module(module)
            implementation = getattr(loaded, attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigReferenceError(f"cannot load plugin {self.descriptor['id']} ({entrypoint}): {exc}; sync its sources and install its runtime dependencies") from exc
        # Python caches modules process-wide. A different registration must not
        # silently reuse an implementation imported from another source tree.
        expected = []
        for candidate in candidates:
            module_path = candidate.joinpath(*module.split("."))
            expected.extend(p.resolve() for p in (module_path.with_suffix(".py"), module_path / "__init__.py") if p.is_file())
        if expected and Path(getattr(loaded, "__file__", "")).resolve() not in expected:
            raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} was imported from a different source; restart the process with the selected registry")
        if isinstance(implementation, type):
            try:
                implementation = implementation()
            except TypeError as exc:
                raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} must be a concrete zero-argument PolicyPlugin class") from exc
        from ovlab_policy_sdk.plugin import PolicyPlugin
        if not isinstance(implementation, PolicyPlugin):
            raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} must inherit PolicyPlugin")
        if getattr(implementation, "api_version", None) != API_VERSION:
            raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} requires unsupported plugin API {getattr(implementation, 'api_version', None)!r}")
        for method in ("validate_policy", "resolve_settings", "create_adapter", "identity"):
            if not callable(getattr(implementation, method, None)):
                raise ConfigCompatibilityError(f"plugin {self.descriptor['id']} is missing API method {method}")
        return implementation


def registered_plugins(repository_root=None):
    root = _root(repository_root)
    document = registry_document(root)
    return [LoadedPolicyPlugin(deepcopy(d), root, document) for d in document["plugins"].values()]


def plugin_for_type(config_type, repository_root=None):
    for plugin in registered_plugins(repository_root):
        if plugin.descriptor["config_type"] == config_type:
            return plugin
    raise ConfigSchemaError(f"unknown policy type {config_type!r}; register its plugin first")


def plugin_for_profile(profile, repository_root=None):
    for plugin in registered_plugins(repository_root):
        if plugin.descriptor.get("deployment", {}).get("profile") == profile:
            return plugin
    raise ConfigSchemaError(f"unknown deployment profile {profile!r}")


def call_plugin(plugin, method, *args, **kwargs):
    """Translate SDK failures at the orchestration boundary."""
    from ovlab_policy_sdk.plugin import (PolicyPluginValidationError, PolicyPluginCompatibilityError, PolicyPluginReferenceError)
    try:
        return getattr(plugin.implementation, method)(*args, **kwargs)
    except PolicyPluginValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
    except PolicyPluginCompatibilityError as exc:
        raise ConfigCompatibilityError(str(exc)) from exc
    except PolicyPluginReferenceError as exc:
        raise ConfigReferenceError(str(exc)) from exc


def plugin_for_document(kind, repository_root=None):
    return next((p for p in registered_plugins(repository_root) if kind in p.descriptor.get("document_kinds", [])), None)
