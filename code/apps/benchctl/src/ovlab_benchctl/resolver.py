"""Explicit experiment composition and typed settings construction."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
from typing import Any

import numpy as np

from ovlab_core.contracts import (
    ActionRepresentation, ActionSpec, GripperConvention, RotationRepresentation,
)
from ovlab_policy_sdk.adapter import _action_specs_equal
from ovlab_metrics import (
    ActionModificationMetricConfig, ActionSequenceMetricConfig, ActionSource, EmptyMetricConfig,
    GripperFlickerMetricConfig, RepeatedNoOpMetricConfig, SuccessRateMetricConfig,
)
from ovlab_runner import (
    ActionExecutionMode, ActionExecutionPolicy, ArtifactStoreSettings, EpisodeErrorPolicy,
    MetricAvailabilityPolicy, TraceRecordingPolicy,
)

from .policy_plugins import with_registry

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from .models import (
    MetricSetSettings, MockBenchmarkSettings, MockPolicySettings, ProtocolSettings, ReportingSettings,
    ResolvedExperimentConfig,
)
from .schema import SCHEMA_VERSION, validate
from .strict_yaml import load
from .config_references import ConfigReferences


def _plain(value):
    if hasattr(value, "items"): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(item) for item in value]
    return value


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "extends": continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class ConfigResolver:
    def __init__(self, config_root: str | Path, *, repository_root: str | Path | None = None, environment=None) -> None:
        self.environment = environment
        self.config_root = Path(config_root).resolve()
        self.repository_root = Path(repository_root).resolve() if repository_root else self.config_root.parent.resolve()
        self.references = ConfigReferences(self.repository_root, self.config_root)

    @staticmethod
    def _inside(path: Path, root: Path, label: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ConfigReferenceError(f"{label} escapes its permitted root: {path}")
        return resolved

    def _root_reference(self, reference: str, label: str) -> Path:
        if not isinstance(reference, str) or not reference:
            raise ConfigReferenceError(f"{label} must be a non-empty configuration reference")
        return self.references.source(reference)

    def _load_composed(self, path: Path, expected_kind: str, stack=(), *, validate_document=True) -> dict[str, Any]:
        path = Path(path).resolve()
        self.references.origin(path)
        if path in stack:
            chain = " -> ".join(str(item) for item in stack + (path,))
            raise ConfigReferenceError(f"extends cycle detected: {chain}")
        raw = load(path)
        expected_version = "ovlab.deployment/v1" if expected_kind == "deployment" else SCHEMA_VERSION
        if raw.get("schema_version") != expected_version:
            raise ConfigSchemaError(f"{path}.schema_version must equal {expected_version!r}")
        if raw.get("kind") != expected_kind:
            raise ConfigSchemaError(f"{path}.kind must equal {expected_kind!r}")
        reference = raw.get("extends")
        if reference is None:
            resolved = raw
        else:
            if not isinstance(reference, str) or not reference:
                raise ConfigSchemaError(f"{path}.extends must be a non-empty relative path")
            parent_path = self.references.parent(path, reference)
            parent = self._load_composed(parent_path, expected_kind, stack + (path,), validate_document=False)
            if "type" in raw and raw["type"] != parent.get("type"):
                raise ConfigSchemaError(f"{path}.type must match its parent")
            resolved = _merge(parent, raw)
        if validate_document:
            validate(resolved, str(path), expected_kind, repository_root=self.repository_root)
        return resolved

    @with_registry
    def load_component(self, reference: str, expected_kind: str) -> dict[str, Any]:
        """Load, inherit, and validate one root-relative component document."""
        if expected_kind == "experiment" and reference.startswith(("experiments/", "examples/")):
            root = self.repository_root / reference.split("/", 1)[0]
            return self._load_composed(self._inside(self.repository_root / reference, root, "experiment"), expected_kind)
        return self._load_composed(self._root_reference(reference, "component"), expected_kind)

    @staticmethod
    def _local_path(value: str, path: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute(): raise ConfigSchemaError(f"{path} must be an absolute path")
        return candidate.resolve()

    @staticmethod
    def _under(root: Path, relative: str, label: str) -> Path:
        path = Path(relative)
        if path.is_absolute(): raise ConfigReferenceError(f"{label} must be relative")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root): raise ConfigReferenceError(f"{label} escapes its resource root")
        return resolved

    @with_registry
    def resolve(
        self,
        experiment: str | Path,
        *,
        local_profile: str | Path,
        execution_profile: str | Path | None = None,
        environment=None,
    ) -> ResolvedExperimentConfig:
        experiment_path = (self.references.source(str(experiment)) if str(experiment).startswith(("plugin:", "module:", "ovlab:"))
                           else Path(experiment))
        if not experiment_path.is_absolute(): experiment_path = self.repository_root / experiment_path
        self.references.origin(experiment_path)
        experiment_doc = self._load_composed(experiment_path, "experiment")

        kinds = {
            "benchmark": "benchmark", "policy": "policy", "metrics": "metric_set", "protocol": "protocol",
            "artifacts": "artifact_store",
        }
        components = {}
        for name, kind in kinds.items():
            path = self._root_reference(experiment_doc["components"][name], f"components.{name}")
            components[name] = self._load_composed(path, kind)
        action_reference = components["policy"]["settings"]["action"]["action_interface"]
        action_path = self._root_reference(action_reference, "policy.settings.action.action_interface")
        action_doc = self._load_composed(action_path, "action_interface")
        registry_path = self._root_reference(experiment_doc["resources"]["registry"], "resources.registry")
        registry = self._load_composed(registry_path, "resource_registry")
        profile_path = (self.references.source(str(local_profile))
                        if str(local_profile).startswith(("ovlab:", "plugin:", "module:")) else Path(local_profile))
        if not profile_path.is_absolute(): profile_path = self.repository_root / profile_path
        profile = load(profile_path.resolve()); validate(profile, str(profile_path), "local_profile")

        paths = {key: self._local_path(value, f"local_profile.paths.{key}") for key, value in profile["paths"].items()}
        devices = dict(profile["devices"])
        local_checkpoint_paths = {
            resource_id: self._local_path(
                entry["local_path"], f"local_profile.resources.checkpoints.{resource_id}.local_path"
            )
            for resource_id, entry in profile.get("resources", {}).get("checkpoints", {}).items()
        }
        values = os.environ if environment is None else environment
        resolved_id = values.get("OVLAB_RESOLVED_CHECKPOINT_ID")
        resolved_path = values.get("OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH")
        if bool(resolved_id) != bool(resolved_path):
            raise ConfigSchemaError(
                "OVLAB_RESOLVED_CHECKPOINT_ID and OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH "
                "must be set together"
            )
        if resolved_path is not None:
            local_checkpoint_paths[str(resolved_id)] = self._local_path(
                resolved_path, "OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH"
            )
        resolved_checkpoints = {}
        policy_settings = components["policy"]["settings"]
        selected_checkpoint_ids = {
            policy_settings[key] for key in ("checkpoint_id", "processor_id") if key in policy_settings
        }
        for resource_id, entry in registry["checkpoints"].items():
            selector = entry.get("revision", "local/main")
            revision = entry.get("revision")
            aggregate = entry.get("expected_sha256")
            if resource_id in selected_checkpoint_ids and selector in {"local/main", "remote/main"}:
                if resource_id == resolved_id:
                    revision = values.get("OVLAB_CHECKPOINT_REVISION")
                    aggregate = values.get("OVLAB_CHECKPOINT_SHA256")
                    if not isinstance(revision, str) or not isinstance(aggregate, str):
                        raise ConfigSchemaError("resolved checkpoint requires OVLAB_CHECKPOINT_REVISION and OVLAB_CHECKPOINT_SHA256")
                else:
                    from .checkpointing import (
                        checkpoint_spec_from_snapshot, default_global_cache, latest_checkpoint_revision,
                    )
                    cache = Path(values.get("OVLAB_GLOBAL_HF_CACHE") or default_global_cache(values)).expanduser().resolve()
                    revision = latest_checkpoint_revision(entry["repo_id"], cache, offline=selector == "local/main")
                    hub = cache / "hub" if (cache / "hub").is_dir() else cache
                    snapshot = hub / ("models--" + entry["repo_id"].replace("/", "--")) / "snapshots" / revision
                    local = local_checkpoint_paths.get(resource_id)
                    if not snapshot.is_dir() and local is not None and local.name == revision:
                        snapshot = local
                    if not snapshot.is_dir():
                        raise ConfigSchemaError(f"checkpoint {resource_id}@{revision} selected by {selector} is not cached; use eval run or deploy run to resolve it")
                    identity = checkpoint_spec_from_snapshot(resource_id, entry["repo_id"], revision, snapshot)
                    aggregate = identity.expected_sha256
                    local_checkpoint_paths[resource_id] = snapshot.resolve()
            resolved = {
                "source": entry["repo_id"],
                "revision": revision, "expected_sha256": aggregate,
            }
            if "artifact" in entry:
                resolved["artifact"] = deepcopy(entry["artifact"])
                resolved["method"] = deepcopy(entry["method"])
                resolved["repo_id"] = entry["repo_id"]
            if resource_id in local_checkpoint_paths:
                resolved["local_path"] = str(local_checkpoint_paths[resource_id])
            resolved_checkpoints[resource_id] = resolved
        resolved_repositories = {}
        for resource_id, entry in registry["repositories"].items():
            resolved_repositories[resource_id] = str(self._under(self.repository_root, entry["path"], f"repository {resource_id}"))
        resources = {
            "local_profile_id": profile["id"], "paths": {key: str(value) for key, value in paths.items()},
            "devices": devices, "checkpoints": resolved_checkpoints, "repositories": resolved_repositories,
        }

        protocol = self._protocol(components["protocol"])
        action_spec = self._action_spec(action_doc)
        execution_profile_doc = None
        renderer_metadata = None
        benchmark_context = None
        benchmark_owner = None
        if components["benchmark"]["type"] == "mock":
            benchmark = self._benchmark(components["benchmark"], protocol, None)
        else:
            from .benchmark_plugins import benchmark_for_document, call_benchmark, peer_context
            from ovlab_core.contracts import BenchmarkResolutionContext
            benchmark_owner = benchmark_for_document(components["benchmark"], self.repository_root)
            profiles = benchmark_owner.descriptor.get("execution_profiles", {})
            configured = experiment_doc.get("deployment", {}).get("renderer")
            frozen_execution = None
            if self.references.is_frozen:
                frozen_execution = json.loads((self.config_root / ".ovlab-bundle.json").read_text()).get("execution_profile")
            reference = execution_profile or frozen_execution or experiment_doc.get("deployment", {}).get("execution_profile") or (profiles.get(configured) if configured else benchmark_owner.descriptor.get("default_execution_profile"))
            if configured and profiles and configured not in profiles and not execution_profile and not experiment_doc.get("deployment", {}).get("execution_profile"):
                raise ConfigCompatibilityError(f"selected benchmark does not support execution profile {configured!r}")
            if reference is not None:
                execution_profile_path = (self._root_reference(str(reference), "execution_profile")
                    if not Path(reference).is_absolute() else Path(reference))
                self.references.origin(execution_profile_path)
                execution_profile_doc = self._load_composed(execution_profile_path, "execution_profile")
                if execution_profile_doc.get("provider") != str(benchmark_owner.ref):
                    raise ConfigCompatibilityError("execution profile belongs to a different benchmark provider")
            context = BenchmarkResolutionContext(protocol.base_seed, protocol.maximum_episode_steps,
                repository_root=self.repository_root, devices=devices, paths={key: str(value) for key,value in paths.items()},
                environment=values, execution={"profile": execution_profile_doc or {},
                    "local": profile.get("execution", {}), "deployment": experiment_doc.get("deployment", {})})
            benchmark = call_benchmark(benchmark_owner, "resolve_settings", components["benchmark"], context)
            benchmark_context = peer_context(benchmark_owner, benchmark)
            runtime_metadata = call_benchmark(benchmark_owner, "execution_metadata", benchmark)
            renderer_metadata = runtime_metadata.get("renderer")
            if not _action_specs_equal(action_spec, benchmark_context.capabilities.action_spec):
                raise ConfigCompatibilityError("policy action interface differs from the benchmark native action interface")
        self._cross_validate_refs(experiment_doc, components, action_path, action_spec, benchmark_context)
        from .policy_plugins import plugin_for_type, call_plugin
        if components["policy"]["type"] != "mock":
            plugin = plugin_for_type(components["policy"]["type"], self.repository_root, provider=components["policy"].get("provider"))
            resources.update(call_plugin(plugin, "runtime_resources", components["policy"], self.repository_root, values))
        policy = self._policy(components["policy"], benchmark_context, action_spec, resolved_checkpoints, devices, environment=values)
        metrics = self._metrics(components["metrics"])
        artifact = self._artifacts(components["artifacts"], paths)
        reporting = ReportingSettings(**experiment_doc.get("reporting", {}))
        if protocol.trace_recording_policy.record_raw_policy_output and not policy.record_raw_output:
            raise ConfigCompatibilityError("protocol requests raw policy output but the policy does not expose it")
        if not components["protocol"]["recording"]["predictions"]:
            raise ConfigCompatibilityError("OVLAB traces always require predictions; recording.predictions must be true")


        scientific_components = deepcopy(components)
        if components["policy"]["type"] != "mock":
            scientific_components["policy"] = call_plugin(plugin, "scientific_policy", components["policy"])
        scientific_components["policy"]["settings"]["action"]["resolved_action_interface"] = action_doc
        scientific = {
            "schema_version": SCHEMA_VERSION, "kind": "scientific_experiment",
            "experiment": experiment_doc["experiment"], "components": scientific_components,
            "resource_registry": registry,
        }
        if components["policy"]["type"] != "mock":
            from .plugin_identity import scientific_source_identities
            scientific["policy_plugin"] = {
                "api_version": plugin.implementation.api_version, "descriptor": plugin.descriptor,
                "sources": scientific_source_identities(
                    self.repository_root, plugin.descriptor.get("sources", []), registry=plugin.registry,
                ),
            }
        moving_selected = sorted(
            resource_id for resource_id in selected_checkpoint_ids
            if registry["checkpoints"][resource_id].get("revision", "local/main") in {"local/main", "remote/main"}
        )
        if moving_selected:
            scientific["resolved_checkpoint_identity"] = {
                resource_id: {
                    key: resolved_checkpoints[resource_id][key]
                    for key in ("source", "revision", "expected_sha256")
                }
                for resource_id in moving_selected
            }
        execution = {"scientific_config": scientific, "resolved_resources": resources}
        if "deployment" in experiment_doc:
            configured_deployment = deepcopy(experiment_doc["deployment"])
            resolved_profile = values.get(
                "OVLAB_DEPLOYMENT_PROFILE", configured_deployment["profile"]
            )
            from .policy_plugins import plugin_for_profile
            plugin_for_profile(resolved_profile, self.repository_root)
            execution["deployment"] = {
                "configured": configured_deployment,
                "resolved": {
                    "profile": resolved_profile,
                    "renderer": renderer_metadata,
                },
            }
        if execution_profile_doc is not None:
            execution["execution_profile"] = execution_profile_doc

        if benchmark_context is not None:
            from .plugin_identity import scientific_source_identities
            from .policy_plugins import registry_document
            identity = benchmark_context.identity
            scientific["benchmark_plugin"] = {"provider": str(benchmark_owner.ref),
                "api_version": benchmark_owner.implementation.api_version,
                "descriptor": benchmark_owner.descriptor,
                "sources": scientific_source_identities(self.repository_root,
                    benchmark_owner.descriptor["sources"], registry=registry_document(self.repository_root)),
                "identity": {"project": identity.plugin_id, "benchmark": identity.benchmark_id, "version": identity.version,
                             "source_revision": identity.source_revision}, "catalog_sha256": benchmark_context.task_set.catalog_sha256}
            execution["benchmark"] = {"provider": str(benchmark_owner.ref), "runtime": runtime_metadata}
        return ResolvedExperimentConfig(
            experiment_doc["experiment"]["id"], benchmark, policy, action_spec, metrics, protocol, artifact, reporting,
            scientific, execution, _hash(scientific), _hash(execution), benchmark_context,
        )

    def _cross_validate_refs(self, experiment, components, action_path, action_spec, benchmark_context=None):
        deployment = experiment.get("deployment")
        if deployment is not None:
            from .policy_plugins import plugin_for_type
            expected_profile = plugin_for_type(components["policy"]["type"], self.repository_root, provider=components["policy"].get("provider")).descriptor.get("deployment", {}).get("profile")
            if expected_profile is None:
                raise ConfigCompatibilityError(
                    f"policy type {components['policy']['type']!r} has no deployment profile"
                )
            if deployment["profile"] != expected_profile:
                raise ConfigCompatibilityError(
                    f"deployment profile {deployment['profile']!r} is incompatible with policy "
                    f"type {components['policy']['type']!r}; expected {expected_profile!r}"
                )
        if benchmark_context is None:
            reference = components["benchmark"]["settings"]["action"]["interface_ref"]
            benchmark_action = self._action_spec(self._load_composed(self._root_reference(reference, "benchmark action"), "action_interface"))
            if not _action_specs_equal(action_spec, benchmark_action):
                raise ConfigCompatibilityError("benchmark action interface differs from the policy interface")
        if components["policy"]["type"] != "mock":
            from .policy_plugins import plugin_for_type, call_plugin
            call_plugin(plugin_for_type(components["policy"]["type"], self.repository_root, provider=components["policy"].get("provider")), "validate_observation", components["policy"], benchmark_context)

    @staticmethod
    def _action_spec(doc):
        try:
            units = doc["units"]
            if isinstance(units, str): units = (units,) * doc["dimension"]
            return ActionSpec(
                doc["dimension"], ActionRepresentation(doc["representation"]), tuple(doc["translation_indices"]),
                tuple(doc["rotation_indices"]), tuple(doc["gripper_indices"]),
                RotationRepresentation(doc["rotation_representation"]), GripperConvention(doc["gripper_convention"]),
                tuple(units), np.asarray(doc["minimum"], dtype=np.float32), np.asarray(doc["maximum"], dtype=np.float32),
                doc["dtype"], float(doc["control_frequency_hz"]),
                {"interface_id": doc["id"]},
            )
        except Exception as exc:
            raise ConfigSchemaError("invalid action interface contract") from exc

    @staticmethod
    def _protocol(doc):
        execution, recording, repro = doc["execution"], doc["recording"], doc["reproducibility"]
        chunks = execution["action_chunks"]
        mode = ActionExecutionMode(chunks["mode"])
        interval = chunks["replan_interval"] if mode is ActionExecutionMode.FIXED_REPLAN_INTERVAL else None
        if mode is not ActionExecutionMode.FIXED_REPLAN_INTERVAL and chunks["replan_interval"] != 1:
            raise ConfigSchemaError("non-fixed action chunk modes require replan_interval: 1")
        trace = TraceRecordingPolicy(
            record_policy_observations=recording["observations"], record_image_arrays=recording["images"],
            record_proprioception=recording["proprioception"], record_raw_policy_output=recording["raw_policy_output"],
            record_evaluation_signals=recording["evaluation_signals"], record_privileged_signals=recording["privileged_signals"],
        )
        return ProtocolSettings(
            execution["rollouts_per_task"], execution["base_seed"], execution["maximum_episode_steps"],
            ActionExecutionPolicy(mode, interval), EpisodeErrorPolicy(execution["episode_errors"]),
            MetricAvailabilityPolicy(execution["unavailable_metrics"]), trace,
            repro["reject_dirty_external_repositories"], repro["require_checkpoint_identity"],
        )

    @staticmethod
    def _benchmark(doc, protocol, renderer):
        settings, obs = doc["settings"], doc["settings"]["observation"]
        if doc["type"] == "mock":
            if settings["maximum_episode_steps"] != protocol.maximum_episode_steps:
                raise ConfigCompatibilityError("mock benchmark and protocol maximum_episode_steps must match")
            return MockBenchmarkSettings(
                settings["task_count"], settings["maximum_episode_steps"], settings["modify_actions"],
                tuple(settings["terminal_outcomes"]), obs["camera"], (obs["height"], obs["width"], 3),
                obs["proprioception"], settings["privileged_signals"]["enabled"],
            )
        raise ConfigCompatibilityError("benchmark settings require its registered SDK provider")

    @with_registry
    def _policy(self, doc, benchmark_context, action_spec, checkpoints, devices, *, environment=None):
        settings = doc["settings"]
        if doc["type"] == "mock":
            return MockPolicySettings(settings["horizon"], settings["deterministic"], settings["input"]["camera"], settings["input"]["proprioception"], action_spec, settings["raw_output"]["enabled"])
        from .policy_plugins import plugin_for_type, call_plugin
        values = dict(os.environ if environment is None else environment)
        resolved = call_plugin(plugin_for_type(doc["type"], self.repository_root, provider=doc.get("provider")), "resolve_settings", doc, benchmark_context, action_spec, checkpoints, devices, repository_root=self.repository_root, environment=values)
        if type(getattr(resolved, "record_raw_output", None)) is not bool:
            raise ConfigCompatibilityError(f"plugin {doc['type']} settings must expose boolean record_raw_output")
        return resolved

    @staticmethod
    def _metrics(doc):
        enabled, configurations = [], {}
        for metric_id, values in doc["plugins"].items():
            if not values["enabled"]: continue
            enabled.append(metric_id)
            if metric_id in ("action.variance", "action.smoothness_1", "action.smoothness_2"):
                config = ActionSequenceMetricConfig(ActionSource(values["action_source"]))
            elif metric_id == "failure.action_modification_rate":
                config = ActionModificationMetricConfig(values["absolute_tolerance"], values["relative_tolerance"])
            elif metric_id == "failure.repeated_no_op_rate":
                config = RepeatedNoOpMetricConfig(ActionSource(values["action_source"]), None, values["norm_threshold"], values["minimum_consecutive_steps"])
            elif metric_id == "failure.gripper_flicker_rate":
                config = GripperFlickerMetricConfig(ActionSource(values["action_source"]), values["activation_threshold"], values["deadband"], values["flicker_window_steps"], values["minimum_dwell_steps"])
            elif metric_id == "task.success_rate": config = SuccessRateMetricConfig()
            else: config = EmptyMetricConfig()
            configurations[metric_id] = config
        return MetricSetSettings(tuple(enabled), tuple(doc["required"]), configurations)

    @staticmethod
    def _artifacts(doc, paths):
        resource = doc["settings"]["root_resource"]
        try: root = paths[resource]
        except KeyError as exc: raise ConfigReferenceError(f"unknown path resource: {resource}") from exc
        return ArtifactStoreSettings(str(root))
