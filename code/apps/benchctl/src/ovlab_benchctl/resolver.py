"""Explicit experiment composition and typed settings construction."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
from typing import Any

import numpy as np

from ovlab_benchmarks.libero import (
    InitialStateSelection, LiberoAdapterSettings, LiberoObservationProfile, resolve_renderer_settings,
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
    LiberoActionCodecConfig, OpenVlaModelSource, OpenVlaRuntimeArtifact, action_specs_match,
    method_descriptor_from_registry,
)
from ovlab_openvla_vanilla import (
    InferenceSynchronization, ModelDType, OpenVlaVanillaSettings,
)
from ovlab_openvla_oft import OpenVlaOftArtifact, OpenVlaOftSettings
from ovlab_runner import (
    ActionExecutionMode, ActionExecutionPolicy, ArtifactStoreSettings, EpisodeErrorPolicy,
    MetricAvailabilityPolicy, TraceRecordingPolicy,
)

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from .models import (
    MetricSetSettings, MockBenchmarkSettings, MockPolicySettings, ProtocolSettings,
    ResolvedExperimentConfig,
)
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

    def resolve(
        self,
        experiment: str | Path,
        *,
        local_profile: str | Path,
        execution_profile: str | Path | None = None,
        environment=None,
    ) -> ResolvedExperimentConfig:
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
        for resource_id, entry in registry["checkpoints"].items():
            resolved = {
                "source": entry["repo_id"],
                "revision": entry["revision"], "expected_sha256": entry["expected_sha256"],
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

        execution_profile_doc = None
        renderer = None
        if components["benchmark"]["type"] == "libero":
            reference = execution_profile or "profiles/libero-bench-egl.yaml"
            execution_profile_path = Path(reference)
            if execution_profile_path.is_absolute():
                execution_profile_path = self._inside(execution_profile_path, self.config_root, "execution_profile")
            else:
                execution_profile_path = self._root_reference(str(execution_profile_path).removeprefix("configs/"), "execution_profile")
            execution_profile_doc = self._load_composed(execution_profile_path, "execution_profile")
            renderer_doc = execution_profile_doc["execution"]["libero"]["renderer"]
            device_id = renderer_doc.get("device_id")
            local_renderer = profile.get("execution", {}).get("libero", {}).get("renderer", {})
            if "device_id" in local_renderer:
                device_id = local_renderer["device_id"]
            if device_id is None:
                device = devices.get("primary_gpu")
                if device is not None:
                    if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
                        raise ConfigCompatibilityError("LIBERO EGL rendering requires primary_gpu: cuda:<index>")
                    device_id = int(device.split(":", 1)[1])
            renderer = resolve_renderer_settings(
                renderer_doc["backend"], device_id, os.environ if environment is None else environment
            )

        action_spec = self._action_spec(components["action_interface"])
        self._cross_validate_refs(experiment_doc, components, component_paths, action_spec)
        protocol = self._protocol(components["protocol"])
        benchmark = self._benchmark(components["benchmark"], protocol, renderer)
        policy = self._policy(components["policy"], components["benchmark"], action_spec, resolved_checkpoints, devices)
        metrics = self._metrics(components["metrics"])
        artifact = self._artifacts(components["artifacts"], paths)
        if protocol.trace_recording_policy.record_raw_policy_output and not policy.record_raw_output:
            raise ConfigCompatibilityError("protocol requests raw policy output but the policy does not expose it")
        if not components["protocol"]["recording"]["predictions"]:
            raise ConfigCompatibilityError("OVLAB traces always require predictions; recording.predictions must be true")
        if components["benchmark"]["type"] == "libero" and not components["benchmark"]["settings"]["privileged_signals"]["enabled"]:
            raise ConfigCompatibilityError("LiberoBenchmarkAdapter always exposes its declared privileged signal registry")

        scientific = {
            "schema_version": SCHEMA_VERSION, "kind": "scientific_experiment",
            "experiment": experiment_doc["experiment"], "components": components,
            "resource_registry": registry,
        }
        execution = {"scientific_config": scientific, "resolved_resources": resources}
        if execution_profile_doc is not None:
            execution["execution_profile"] = execution_profile_doc
            execution["libero"] = {"renderer": renderer.as_dict()}
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
        if components["benchmark"]["type"] == "libero" and not action_specs_match(action_spec, libero_action_spec()):
            raise ConfigCompatibilityError("action interface differs from LiberoBenchmarkAdapter's verified ActionSpec")
        policy_input = components["policy"]["settings"]["input"]
        observation = components["benchmark"]["settings"]["observation"]
        benchmark_camera = (
            observation["cameras"]["primary"]["canonical_name"]
            if components["benchmark"]["type"] == "libero"
            else observation["camera"]
        )
        if components["policy"]["type"] == "openvla_oft":
            cameras = policy_input["cameras"]
            supplied = {item["canonical_name"] for item in observation["cameras"].values()}
            if set(cameras.values()) != supplied:
                raise ConfigCompatibilityError("OFT image inputs differ from the benchmark observation interface")
            if policy_input["proprioception"] != "robot.proprioception" or observation["profile"] != "native_oft":
                raise ConfigCompatibilityError("OFT requires the native LIBERO proprioception observation profile")
        elif policy_input["camera"] != benchmark_camera:
            raise ConfigCompatibilityError("policy input camera is not supplied by the benchmark observation interface")
        if components["benchmark"]["type"] == "mock":
            expected_proprio = components["policy"]["settings"]["input"]["proprioception"]
            if expected_proprio is not None and expected_proprio != observation["proprioception"]:
                raise ConfigCompatibilityError("policy proprioception is not supplied by the benchmark observation interface")

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
        if renderer is None:
            raise ConfigCompatibilityError("LIBERO benchmark requires an execution renderer profile")
        suite = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
                 "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-10"}[settings["suite"]]
        task_indices = None if settings["task_indices"] == "all" else tuple(settings["task_indices"])
        camera_names = [obs["cameras"]["primary"]["native_name"]]
        if "wrist" in obs["cameras"]:
            camera_names.append(obs["cameras"]["wrist"]["native_name"])
        return LiberoAdapterSettings(
            suite_names=(suite,), task_indices=task_indices,
            camera_names=tuple(camera_names), camera_width=obs["width"], camera_height=obs["height"],
            observation_profile=LiberoObservationProfile(obs["profile"]), maximum_episode_steps=protocol.maximum_episode_steps,
            initialization_settling_steps=settings["initialization"]["settling_steps"],
            initial_state_selection=InitialStateSelection(settings["initialization"]["state_selection"]),
            base_seed=protocol.base_seed, renderer=renderer,
        )

    @staticmethod
    def _policy(doc, benchmark_doc, action_spec, checkpoints, devices):
        settings = doc["settings"]
        if doc["type"] == "mock":
            return MockPolicySettings(
                settings["horizon"], settings["deterministic"], settings["input"]["camera"],
                settings["input"]["proprioception"], action_spec, settings["raw_output"]["enabled"],
            )
        runtime = settings["runtime"]
        if not runtime["local_files_only"]:
            raise ConfigCompatibilityError("OpenVLA production configuration requires local_files_only: true")
        try: model, processor = checkpoints[settings["checkpoint_id"]], checkpoints[settings["processor_id"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown checkpoint resource: {exc.args[0]}") from exc
        try: device = devices[runtime["device_resource"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown device resource: {runtime['device_resource']}") from exc
        obs = benchmark_doc["settings"]["observation"]
        source = lambda item: OpenVlaModelSource(
            item["source"], item["revision"], item["expected_sha256"], item.get("local_path")
        )
        if doc["type"] == "openvla_oft":
            if settings["checkpoint_id"] != settings["processor_id"]:
                raise ConfigCompatibilityError("OFT model and processor resources must be identical")
            if benchmark_doc["settings"]["suite"] != "libero_10" or obs["profile"] != "native_oft":
                raise ConfigCompatibilityError("native Gate E OFT requires LIBERO-10 and the native_oft profile")
            registry_entry = {
                "repo_id": model["source"], "revision": model["revision"],
                "expected_sha256": model["expected_sha256"], "artifact": model["artifact"],
                "method": model["method"],
            }
            artifact = OpenVlaOftArtifact.from_registry_entry(settings["checkpoint_id"], registry_entry)
            cameras = settings["input"]["cameras"]
            return OpenVlaOftSettings(
                source(model), artifact, unnorm_key=settings["unnorm_key"],
                primary_camera_name=cameras["primary"], wrist_camera_name=cameras["wrist"],
                proprioception_name=settings["input"]["proprioception"],
                input_image_shape=(obs["height"], obs["width"], 3),
                device=device, attention_implementation=runtime["attention_implementation"],
                target_action_spec=action_spec, record_raw_output=settings["raw_output"]["enabled"],
                metadata={"execution_strategy": "open_loop_chunk"},
            )
        method_descriptor = None
        runtime_artifact = None
        if doc["type"] == "openvla_lora_merged":
            if settings["checkpoint_id"] != settings["processor_id"]:
                raise ConfigCompatibilityError("merged LoRA model and processor resources must be identical")
            if "artifact" not in model or "method" not in model:
                raise ConfigCompatibilityError("merged LoRA policy requires registered artifact and method identity")
            registry_entry = {
                "repo_id": model["source"],
                "revision": model["revision"],
                "expected_sha256": model["expected_sha256"],
                "artifact": model["artifact"],
                "method": model["method"],
            }
            method_descriptor = method_descriptor_from_registry(registry_entry)
            runtime_artifact = OpenVlaRuntimeArtifact.from_registry_entry(
                settings["checkpoint_id"], registry_entry
            )
            if settings["unnorm_key"] != "libero_10":
                raise ConfigCompatibilityError("official merged LoRA reference requires unnorm_key=libero_10")
            if benchmark_doc["settings"]["suite"] != "libero_10":
                raise ConfigCompatibilityError("merged LIBERO-10 LoRA is not a cross-suite acceptance policy")
        elif "artifact" in model or "method" in model:
            raise ConfigCompatibilityError("merged LoRA full weights cannot be classified as OpenVLA Vanilla")
        kwargs = {}
        if method_descriptor is not None:
            kwargs["method_descriptor"] = method_descriptor
            kwargs["runtime_artifact"] = runtime_artifact
        return OpenVlaVanillaSettings(
            source(model), settings["unnorm_key"], processor=source(processor),
            canonical_camera_name=settings["input"]["camera"], input_image_shape=(obs["height"], obs["width"], 3),
            device=device, model_dtype=ModelDType(runtime["dtype"]),
            attention_implementation=runtime["attention_implementation"], local_files_only=runtime["local_files_only"],
            trust_remote_code=runtime["trust_remote_code"], deterministic_inference=runtime["deterministic"],
            target_action_spec=action_spec, action_codec=LiberoActionCodecConfig(),
            synchronization=InferenceSynchronization.IF_CUDA if runtime["synchronize_inference"] else InferenceSynchronization.NONE,
            record_raw_output=settings["raw_output"]["enabled"],
            **kwargs,
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
