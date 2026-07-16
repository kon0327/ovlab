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
    MetricAvailabilityPolicy, TraceRecordingPolicy,
)

from .errors import ResolvedConfigWriteError
from .strict_yaml import dumps


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
    benchmark_settings: LiberoAdapterSettings
    policy_settings: OpenVlaVanillaSettings
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
