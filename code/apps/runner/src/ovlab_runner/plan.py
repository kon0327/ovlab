"""Immutable experiment and recording plans with deterministic hashes."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
import hashlib
import json

from ovlab_core.contracts import Metadata, RunContext, TaskId, normalize_metadata
from ovlab_metrics import canonical_config

from .errors import RunnerError


class ActionExecutionMode(str, Enum):
    RECEDING_HORIZON = "receding_horizon"
    OPEN_LOOP_CHUNK = "open_loop_chunk"
    FIXED_REPLAN_INTERVAL = "fixed_replan_interval"


class MetricAvailabilityPolicy(str, Enum):
    ALLOW_UNAVAILABLE = "allow_unavailable"
    REQUIRE_SELECTED = "require_selected"


class EpisodeErrorPolicy(str, Enum):
    STOP_RUN = "stop_run"
    CONTINUE_TASK = "continue_task"
    CONTINUE_RUN = "continue_run"


@dataclass(frozen=True, slots=True)
class ActionExecutionPolicy:
    mode: ActionExecutionMode = ActionExecutionMode.RECEDING_HORIZON
    replan_interval: int | None = None

    def __post_init__(self):
        if not isinstance(self.mode, ActionExecutionMode):
            raise RunnerError("mode must be an ActionExecutionMode")
        if self.mode is ActionExecutionMode.FIXED_REPLAN_INTERVAL:
            if isinstance(self.replan_interval, bool) or not isinstance(self.replan_interval, int) or self.replan_interval < 1:
                raise RunnerError("fixed replan interval must be a positive integer")
        elif self.replan_interval is not None:
            raise RunnerError("replan_interval is valid only for FIXED_REPLAN_INTERVAL")


@dataclass(frozen=True, slots=True)
class TraceRecordingPolicy:
    record_policy_observations: bool = True
    record_image_arrays: bool = True
    record_proprioception: bool = True
    record_raw_policy_output: bool = True
    record_evaluation_signals: bool = True
    record_privileged_signals: bool = False
    image_sampling_stride: int = 1

    def __post_init__(self):
        if isinstance(self.image_sampling_stride, bool) or self.image_sampling_stride < 1:
            raise RunnerError("image_sampling_stride must be positive")

    def canonical(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def hash(self):
        return _hash(self.canonical())


@dataclass(frozen=True, slots=True)
class ArtifactStoreSettings:
    root: str = "runs"

    def __post_init__(self):
        if not isinstance(self.root, str) or not self.root.strip():
            raise RunnerError("artifact root must be non-empty")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    run_context: RunContext
    selected_task_ids: tuple[TaskId, ...]
    rollout_count_per_task: int
    base_episode_seed: int
    default_maximum_episode_steps: int
    action_execution_policy: ActionExecutionPolicy = field(default_factory=ActionExecutionPolicy)
    enabled_metric_ids: tuple[str, ...] = ()
    metric_configurations: dict = field(default_factory=dict)
    required_metric_ids: tuple[str, ...] = ()
    unavailable_metric_policy: MetricAvailabilityPolicy = MetricAvailabilityPolicy.ALLOW_UNAVAILABLE
    episode_error_policy: EpisodeErrorPolicy = EpisodeErrorPolicy.STOP_RUN
    trace_recording_policy: TraceRecordingPolicy = field(default_factory=TraceRecordingPolicy)
    artifact_store_settings: ArtifactStoreSettings = field(default_factory=ArtifactStoreSettings)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        tasks = tuple(self.selected_task_ids)
        metrics = tuple(self.enabled_metric_ids)
        required = tuple(self.required_metric_ids)
        if not tasks or any(not isinstance(task, TaskId) for task in tasks) or len(tasks) != len(set(tasks)):
            raise RunnerError("selected task IDs must be non-empty, typed, and unique")
        if self.rollout_count_per_task <= 0 or self.default_maximum_episode_steps <= 0 or self.base_episode_seed < 0:
            raise RunnerError("rollout count and maximum steps must be positive; seed must be non-negative")
        if any(not value.strip() for value in metrics) or len(metrics) != len(set(metrics)):
            raise RunnerError("enabled metric IDs must be non-empty and unique")
        if len(required) != len(set(required)) or not set(required) <= set(metrics):
            raise RunnerError("required metrics must be unique enabled metrics")
        configs = dict(self.metric_configurations)
        if not set(configs) <= set(metrics):
            raise RunnerError("metric configuration keys must be enabled metrics")
        object.__setattr__(self, "selected_task_ids", tasks)
        object.__setattr__(self, "enabled_metric_ids", metrics)
        object.__setattr__(self, "required_metric_ids", required)
        object.__setattr__(self, "metric_configurations", MappingProxyType(configs))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))

    def canonical(self) -> dict:
        run = self.run_context
        return {
            "run_context": {"run_id": str(run.run_id), "created_wall_time_utc_ns": run.created_wall_time_utc_ns, "experiment_name": run.experiment_name, "seed": run.seed, "contract_version": run.contract_version, "metadata": _plain(run.metadata)},
            "selected_task_ids": [str(task) for task in self.selected_task_ids],
            "rollout_count_per_task": self.rollout_count_per_task,
            "base_episode_seed": self.base_episode_seed,
            "default_maximum_episode_steps": self.default_maximum_episode_steps,
            "action_execution_policy": {"mode": self.action_execution_policy.mode.value, "replan_interval": self.action_execution_policy.replan_interval},
            "enabled_metric_ids": list(self.enabled_metric_ids),
            "metric_configurations": {key: canonical_config(value) for key, value in sorted(self.metric_configurations.items())},
            "required_metric_ids": list(self.required_metric_ids),
            "unavailable_metric_policy": self.unavailable_metric_policy.value,
            "episode_error_policy": self.episode_error_policy.value,
            "trace_recording_policy": self.trace_recording_policy.canonical(),
            "artifact_store_settings": {"root": self.artifact_store_settings.root},
            "metadata": _plain(self.metadata),
        }

    @property
    def hash(self) -> str:
        return _hash(self.canonical())

    def episode_seed(self, task_id: TaskId, task_order_index: int, rollout_index: int) -> int:
        payload = f"{self.base_episode_seed}\0{task_order_index}\0{task_id}\0{rollout_index}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
