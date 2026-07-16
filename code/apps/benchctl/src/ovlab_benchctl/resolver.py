"""Explicit experiment composition and typed settings construction."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

from ovlab_benchmarks.libero import (
    InitialStateSelection, LiberoAdapterSettings, LiberoObservationProfile, LiberoRenderMode,
)
from ovlab_benchmarks.libero.actions import libero_action_spec
from ovlab_core.contracts import (
    ActionRepresentation, ActionSpec, GripperConvention, RotationRepresentation,
)
from ovlab_metrics import (
    ActionModificationMetricConfig, ActionSequenceMetricConfig, ActionSource, EmptyMetricConfig,
    GripperFlickerMetricConfig, RepeatedNoOpMetricConfig, SuccessRateMetricConfig,
)
from ovlab_openvla_common import (
    LiberoActionCodecConfig, OpenVlaModelSource, action_specs_match,
)
from ovlab_openvla_vanilla import (
    InferenceSynchronization, ModelDType, OpenVlaVanillaSettings,
)
from ovlab_runner import (
    ActionExecutionMode, ActionExecutionPolicy, ArtifactStoreSettings, EpisodeErrorPolicy,
    MetricAvailabilityPolicy, TraceRecordingPolicy,
)

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from .models import MetricSetSettings, ProtocolSettings, ResolvedExperimentConfig
from .schema import SCHEMA_VERSION, validate
from .strict_yaml import load


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
    def __init__(self, config_root: str | Path, *, repository_root: str | Path | None = None) -> None:
        self.config_root = Path(config_root).resolve()
        self.repository_root = Path(repository_root).resolve() if repository_root else self.config_root.parent.resolve()
        if not self.config_root.is_dir(): raise ConfigReferenceError(f"config root does not exist: {self.config_root}")

    @staticmethod
    def _inside(path: Path, root: Path, label: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ConfigReferenceError(f"{label} escapes its permitted root: {path}")
        return resolved

    def _root_reference(self, reference: str, label: str) -> Path:
        if not isinstance(reference, str) or not reference:
            raise ConfigReferenceError(f"{label} must be a non-empty relative path")
        path = Path(reference)
        if path.is_absolute(): raise ConfigReferenceError(f"{label} must be relative to configs/")
        return self._inside(self.config_root / path, self.config_root, label)

    def _load_composed(self, path: Path, expected_kind: str, stack=()) -> dict[str, Any]:
        path = self._inside(path, self.config_root, "configuration")
        if path in stack:
            chain = " -> ".join(str(item) for item in stack + (path,))
            raise ConfigReferenceError(f"extends cycle detected: {chain}")
        raw = load(path)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ConfigSchemaError(f"{path}.schema_version must equal {SCHEMA_VERSION!r}")
        if raw.get("kind") != expected_kind:
            raise ConfigSchemaError(f"{path}.kind must equal {expected_kind!r}")
        reference = raw.get("extends")
        if reference is None:
            resolved = raw
        else:
            if not isinstance(reference, str) or not reference:
                raise ConfigSchemaError(f"{path}.extends must be a non-empty relative path")
            parent_path = self._inside(path.parent / reference, self.config_root, "extends")
            parent = self._load_composed(parent_path, expected_kind, stack + (path,))
            if raw.get("type") != parent.get("type"):
                raise ConfigSchemaError(f"{path}.type must match its parent")
            resolved = _merge(parent, raw)
        validate(resolved, str(path), expected_kind)
        return resolved

    def load_component(self, reference: str, expected_kind: str) -> dict[str, Any]:
        """Load, inherit, and validate one root-relative component document."""
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

    def resolve(self, experiment: str | Path, *, local_profile: str | Path) -> ResolvedExperimentConfig:
        experiment_path = Path(experiment)
        if not experiment_path.is_absolute(): experiment_path = self.repository_root / experiment_path
        experiment_path = self._inside(experiment_path, self.config_root, "experiment")
        experiment_doc = self._load_composed(experiment_path, "experiment")

        kinds = {
            "benchmark": "benchmark", "policy": "policy", "metrics": "metric_set", "protocol": "protocol",
            "action_interface": "action_interface", "artifacts": "artifact_store",
        }
        components, component_paths = {}, {}
        for name, kind in kinds.items():
            path = self._root_reference(experiment_doc["components"][name], f"components.{name}")
            component_paths[name] = path
            components[name] = self._load_composed(path, kind)
        registry_path = self._root_reference(experiment_doc["resources"]["registry"], "resources.registry")
        registry = self._load_composed(registry_path, "resource_registry")
        profile_path = Path(local_profile)
        if not profile_path.is_absolute(): profile_path = self.repository_root / profile_path
        profile = load(profile_path.resolve()); validate(profile, str(profile_path), "local_profile")

        paths = {key: self._local_path(value, f"local_profile.paths.{key}") for key, value in profile["paths"].items()}
        devices = dict(profile["devices"])
        resolved_checkpoints = {}
        for resource_id, entry in registry["checkpoints"].items():
            resolved_checkpoints[resource_id] = {
                "path": str(self._under(paths["checkpoint_root"], entry["relative_path"], f"checkpoint {resource_id}")),
                "expected_revision": entry["expected_revision"], "expected_sha256": entry["expected_sha256"],
            }
        resolved_repositories = {}
        for resource_id, entry in registry["repositories"].items():
            resolved_repositories[resource_id] = str(self._under(self.repository_root, entry["path"], f"repository {resource_id}"))
        resources = {
            "local_profile_id": profile["id"], "paths": {key: str(value) for key, value in paths.items()},
            "devices": devices, "checkpoints": resolved_checkpoints, "repositories": resolved_repositories,
        }

        action_spec = self._action_spec(components["action_interface"])
        self._cross_validate_refs(experiment_doc, components, component_paths, action_spec)
        protocol = self._protocol(components["protocol"])
        benchmark = self._benchmark(components["benchmark"], protocol, devices)
        policy = self._policy(components["policy"], components["benchmark"], action_spec, resolved_checkpoints, devices)
        metrics = self._metrics(components["metrics"])
        artifact = self._artifacts(components["artifacts"], paths)
        if protocol.trace_recording_policy.record_raw_policy_output and not policy.record_raw_output:
            raise ConfigCompatibilityError("protocol requests raw policy output but the policy does not expose it")
        if not components["protocol"]["recording"]["predictions"]:
            raise ConfigCompatibilityError("OVLAB traces always require predictions; recording.predictions must be true")
        if not components["benchmark"]["settings"]["privileged_signals"]["enabled"]:
            raise ConfigCompatibilityError("LiberoBenchmarkAdapter always exposes its declared privileged signal registry")

        scientific = {
            "schema_version": SCHEMA_VERSION, "kind": "scientific_experiment",
            "experiment": experiment_doc["experiment"], "components": components,
            "resource_registry": registry,
        }
        execution = {"scientific_config": scientific, "resolved_resources": resources}
        return ResolvedExperimentConfig(
            experiment_doc["experiment"]["id"], benchmark, policy, action_spec, metrics, protocol, artifact,
            scientific, execution, _hash(scientific), _hash(execution),
        )

    def _cross_validate_refs(self, experiment, components, paths, action_spec):
        expected = paths["action_interface"]
        for owner in ("benchmark", "policy"):
            reference = components[owner]["settings"]["action"]["interface_ref"]
            actual = self._root_reference(reference, f"{owner}.settings.action.interface_ref")
            if actual != expected:
                raise ConfigCompatibilityError(f"{owner} action interface differs from the experiment interface")
        if not action_specs_match(action_spec, libero_action_spec()):
            raise ConfigCompatibilityError("action interface differs from LiberoBenchmarkAdapter's verified ActionSpec")
        camera = components["policy"]["settings"]["input"]["camera"]
        benchmark_camera = components["benchmark"]["settings"]["observation"]["cameras"]["primary"]["canonical_name"]
        if camera != benchmark_camera:
            raise ConfigCompatibilityError("policy input camera is not supplied by the benchmark observation interface")

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
    def _benchmark(doc, protocol, devices):
        settings, obs = doc["settings"], doc["settings"]["observation"]
        render = settings["rendering"]
        try: device = devices[render["gpu_resource"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown device resource: {render['gpu_resource']}") from exc
        if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
            raise ConfigCompatibilityError("LIBERO headless rendering requires a cuda:<index> device resource")
        suite = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
                 "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-10"}[settings["suite"]]
        task_indices = None if settings["task_indices"] == "all" else tuple(settings["task_indices"])
        return LiberoAdapterSettings(
            suite_names=(suite,), task_indices=task_indices,
            camera_names=(obs["cameras"]["primary"]["native_name"],), camera_width=obs["width"], camera_height=obs["height"],
            observation_profile=LiberoObservationProfile(obs["profile"]), maximum_episode_steps=protocol.maximum_episode_steps,
            initialization_settling_steps=settings["initialization"]["settling_steps"],
            initial_state_selection=InitialStateSelection(settings["initialization"]["state_selection"]),
            base_seed=protocol.base_seed, render_mode=LiberoRenderMode(render["mode"]),
            render_gpu_device_id=int(device.split(":", 1)[1]),
        )

    @staticmethod
    def _policy(doc, benchmark_doc, action_spec, checkpoints, devices):
        settings, runtime = doc["settings"], doc["settings"]["runtime"]
        if not runtime["local_files_only"]:
            raise ConfigCompatibilityError("OpenVLA production configuration requires local_files_only: true")
        try: model, processor = checkpoints[settings["checkpoint_id"]], checkpoints[settings["processor_id"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown checkpoint resource: {exc.args[0]}") from exc
        try: device = devices[runtime["device_resource"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown device resource: {runtime['device_resource']}") from exc
        obs = benchmark_doc["settings"]["observation"]
        source = lambda item: OpenVlaModelSource(item["path"], item["expected_revision"], item["expected_sha256"])
        return OpenVlaVanillaSettings(
            source(model), settings["unnorm_key"], processor=source(processor),
            canonical_camera_name=settings["input"]["camera"], input_image_shape=(obs["height"], obs["width"], 3),
            device=device, model_dtype=ModelDType(runtime["dtype"]),
            attention_implementation=runtime["attention_implementation"], local_files_only=runtime["local_files_only"],
            trust_remote_code=runtime["trust_remote_code"], deterministic_inference=runtime["deterministic"],
            target_action_spec=action_spec, action_codec=LiberoActionCodecConfig(),
            synchronization=InferenceSynchronization.IF_CUDA if runtime["synchronize_inference"] else InferenceSynchronization.NONE,
            record_raw_output=settings["raw_output"]["enabled"],
        )

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
