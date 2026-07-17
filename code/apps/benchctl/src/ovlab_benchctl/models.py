"""Immutable typed results produced by configuration composition."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import os
import uuid

from ovlab_benchmarks.libero import LiberoAdapterSettings
from ovlab_core.contracts import ActionSpec, Metadata, normalize_metadata
from ovlab_openvla_vanilla import OpenVlaVanillaSettings
from ovlab_runner import (
    ActionExecutionPolicy, ArtifactStoreSettings, EpisodeErrorPolicy,
    ExperimentPlan, MetricAvailabilityPolicy, RunConfigurationSnapshot, TraceRecordingPolicy,
)

from .errors import ResolvedConfigWriteError
from .strict_yaml import dumps


@dataclass(frozen=True, slots=True)
class MockBenchmarkSettings:
    task_count: int
    maximum_episode_steps: int
    modify_actions: bool
    terminal_outcomes: tuple[str, ...]
    camera_name: str
    image_shape: tuple[int, int, int]
    proprioception_name: str
    privileged_signals_enabled: bool

    def __post_init__(self) -> None:
        for name in ("task_count", "maximum_episode_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.terminal_outcomes) != self.task_count:
            raise ValueError("terminal_outcomes must contain one outcome per mock task")
        if any(value not in {"success", "failure", "time_limit"} for value in self.terminal_outcomes):
            raise ValueError("mock terminal outcomes must be success, failure, or time_limit")
        if len(self.image_shape) != 3 or any(type(value) is not int or value <= 0 for value in self.image_shape):
            raise ValueError("image_shape must contain three positive integers")
        for name in ("camera_name", "proprioception_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("modify_actions", "privileged_signals_enabled"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class MockPolicySettings:
    horizon: int
    deterministic: bool
    camera_name: str
    proprioception_name: str | None
    action_spec: ActionSpec
    raw_output_enabled: bool

    @property
    def record_raw_output(self) -> bool:
        """Expose the common policy recording capability used by the resolver."""
        return self.raw_output_enabled

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(self.deterministic) is not bool or type(self.raw_output_enabled) is not bool:
            raise TypeError("deterministic and raw_output_enabled must be booleans")
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise ValueError("camera_name must be a non-empty string")
        if self.proprioception_name is not None and (
            not isinstance(self.proprioception_name, str) or not self.proprioception_name
        ):
            raise ValueError("proprioception_name must be a non-empty string or None")
        if not isinstance(self.action_spec, ActionSpec):
            raise TypeError("action_spec must be an ActionSpec")


@dataclass(frozen=True, slots=True)
class MetricSetSettings:
    enabled_metric_ids: tuple[str, ...]
    required_metric_ids: tuple[str, ...]
    configurations: Mapping[str, object]

    def __post_init__(self) -> None:
        enabled, required, configurations = tuple(self.enabled_metric_ids), tuple(self.required_metric_ids), dict(self.configurations)
        if not enabled or any(not isinstance(item, str) or not item for item in enabled) or len(enabled) != len(set(enabled)):
            raise ValueError("enabled_metric_ids must contain unique non-empty strings")
        if any(not isinstance(item, str) or not item for item in required) or len(required) != len(set(required)):
            raise ValueError("required_metric_ids must contain unique non-empty strings")
        if not set(required) <= set(enabled): raise ValueError("required metrics must be enabled")
        if set(configurations) != set(enabled): raise ValueError("every enabled metric must have one typed configuration")
        object.__setattr__(self, "enabled_metric_ids", enabled)
        object.__setattr__(self, "required_metric_ids", required)
        object.__setattr__(self, "configurations", MappingProxyType(configurations))


@dataclass(frozen=True, slots=True)
class ProtocolSettings:
    rollouts_per_task: int
    base_seed: int
    maximum_episode_steps: int
    action_execution_policy: ActionExecutionPolicy
    episode_error_policy: EpisodeErrorPolicy
    unavailable_metric_policy: MetricAvailabilityPolicy
    trace_recording_policy: TraceRecordingPolicy
    reject_dirty_external_repositories: bool
    require_checkpoint_identity: bool

    def __post_init__(self) -> None:
        for name in ("rollouts_per_task", "maximum_episode_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0: raise ValueError(f"{name} must be a positive integer")
        if type(self.base_seed) is not int or self.base_seed < 0: raise ValueError("base_seed must be non-negative")
        for name in ("reject_dirty_external_repositories", "require_checkpoint_identity"):
            if type(getattr(self, name)) is not bool: raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ResolvedExperimentConfig:
    experiment_id: str
    benchmark_settings: LiberoAdapterSettings | MockBenchmarkSettings
    policy_settings: OpenVlaVanillaSettings | MockPolicySettings
    action_spec: ActionSpec
    metric_settings: MetricSetSettings
    protocol_settings: ProtocolSettings
    artifact_settings: ArtifactStoreSettings
    scientific_config: Metadata
    execution_config: Metadata
    scientific_config_hash: str
    execution_config_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        for name in ("scientific_config_hash", "execution_config_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "scientific_config", normalize_metadata(self.scientific_config, type(self).__name__, "scientific_config"))
        object.__setattr__(self, "execution_config", normalize_metadata(self.execution_config, type(self).__name__, "execution_config"))

    def document(self) -> dict[str, object]:
        return {
            "schema_version": "0.1.0",
            "kind": "resolved_experiment",
            "experiment_id": self.experiment_id,
            "scientific_config_hash": self.scientific_config_hash,
            "execution_config_hash": self.execution_config_hash,
            "scientific_config": self.scientific_config,
            "execution_config": self.execution_config,
        }

    def configuration_snapshot(self) -> RunConfigurationSnapshot:
        return RunConfigurationSnapshot(
            dumps(self.scientific_config), dumps(self.document()),
            self.scientific_config_hash, self.execution_config_hash,
        )

    def create_plan(self, run_context, selected_task_ids) -> ExperimentPlan:
        protocol = self.protocol_settings
        return ExperimentPlan(
            run_context=run_context,
            selected_task_ids=tuple(selected_task_ids),
            rollout_count_per_task=protocol.rollouts_per_task,
            base_episode_seed=protocol.base_seed,
            default_maximum_episode_steps=protocol.maximum_episode_steps,
            action_execution_policy=protocol.action_execution_policy,
            enabled_metric_ids=self.metric_settings.enabled_metric_ids,
            metric_configurations=dict(self.metric_settings.configurations),
            required_metric_ids=self.metric_settings.required_metric_ids,
            unavailable_metric_policy=protocol.unavailable_metric_policy,
            episode_error_policy=protocol.episode_error_policy,
            trace_recording_policy=protocol.trace_recording_policy,
            artifact_store_settings=self.artifact_settings,
            metadata={
                "experiment_id": self.experiment_id,
                "scientific_config_hash": self.scientific_config_hash,
                "execution_config_hash": self.execution_config_hash,
            },
        )

    def write(self, destination: str | Path) -> Path:
        target = Path(destination)
        if target.is_dir(): target = target / "resolved_config.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        payload = dumps(self.document()).encode("utf-8")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ResolvedConfigWriteError(f"resolved configuration already exists: {target}") from exc
        except OSError as exc:
            raise ResolvedConfigWriteError(f"cannot write resolved configuration: {target}") from exc
        finally:
            try: temporary.unlink()
            except FileNotFoundError: pass
        return target
