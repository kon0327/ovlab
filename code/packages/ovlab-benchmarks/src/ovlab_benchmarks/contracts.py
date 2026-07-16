"""Transport-neutral benchmark task, reset, request, and step contracts."""

from dataclasses import dataclass, field

import numpy as np

from ovlab_core.contracts import (
    EpisodeContext,
    ExecutedAction,
    Metadata,
    PolicyObservation,
    PredictionId,
    SignalValue,
    StepContext,
    TaskId,
    immutable_numeric_array,
    normalize_metadata,
)
from ovlab_core.contracts.errors import validation_error
from ovlab_core.contracts.time import validate_timestamp_ns


def _non_empty(value: str, contract: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(contract, field_name, "must not be empty or whitespace-only")


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    suite_name: str
    task_id: TaskId
    task_name: str
    task_index: int
    natural_language_instruction: str
    maximum_steps: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for field_name in ("suite_name", "task_name", "natural_language_instruction"):
            _non_empty(getattr(self, field_name), contract, field_name)
        if not isinstance(self.task_id, TaskId):
            raise validation_error(contract, "task_id", "must be a TaskId")
        if isinstance(self.task_index, bool) or not isinstance(self.task_index, int) or self.task_index < 0:
            raise validation_error(contract, "task_index", "must be a non-negative integer")
        if isinstance(self.maximum_steps, bool) or not isinstance(self.maximum_steps, int) or self.maximum_steps <= 0:
            raise validation_error(contract, "maximum_steps", "must be a positive integer")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class BenchmarkResetResult:
    episode_context: EpisodeContext
    initial_observation: PolicyObservation
    evaluation_signals: tuple[SignalValue, ...]
    timestamp_ns: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.episode_context, EpisodeContext):
            raise validation_error(contract, "episode_context", "must be an EpisodeContext")
        if not isinstance(self.initial_observation, PolicyObservation):
            raise validation_error(contract, "initial_observation", "must be a PolicyObservation")
        signals = tuple(self.evaluation_signals)
        if any(not isinstance(value, SignalValue) for value in signals):
            raise validation_error(contract, "evaluation_signals", "must contain SignalValue values")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        object.__setattr__(self, "evaluation_signals", signals)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class BenchmarkActionRequest:
    step_context: StepContext
    prediction_id: PredictionId
    chunk_index: int
    requested_action: np.ndarray
    timestamp_ns: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.step_context, StepContext):
            raise validation_error(contract, "step_context", "must be a StepContext")
        if not isinstance(self.prediction_id, PredictionId):
            raise validation_error(contract, "prediction_id", "must be a PredictionId")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise validation_error(contract, "chunk_index", "must be a non-negative integer")
        action = immutable_numeric_array(self.requested_action, contract, "requested_action", ndim=1)
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        object.__setattr__(self, "requested_action", action)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class BenchmarkStepResult:
    step_context: StepContext
    executed_action: ExecutedAction
    next_observation: PolicyObservation | None
    evaluation_signals: tuple[SignalValue, ...]
    reward: float | None
    terminated: bool
    truncated: bool
    success: bool | None
    timestamp_ns: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.step_context, StepContext):
            raise validation_error(contract, "step_context", "must be a StepContext")
        if not isinstance(self.executed_action, ExecutedAction):
            raise validation_error(contract, "executed_action", "must be an ExecutedAction")
        if self.executed_action.step_id != self.step_context.step_id:
            raise validation_error(contract, "executed_action", "step_id must match step_context")
        if self.next_observation is not None and not isinstance(self.next_observation, PolicyObservation):
            raise validation_error(contract, "next_observation", "must be a PolicyObservation or None")
        signals = tuple(self.evaluation_signals)
        if any(not isinstance(value, SignalValue) for value in signals):
            raise validation_error(contract, "evaluation_signals", "must contain SignalValue values")
        if self.reward is not None:
            if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)) or not np.isfinite(self.reward):
                raise validation_error(contract, "reward", "must be a finite number or None")
        for field_name in ("terminated", "truncated"):
            if not isinstance(getattr(self, field_name), bool):
                raise validation_error(contract, field_name, "must be a boolean")
        if self.success is not None and not isinstance(self.success, bool):
            raise validation_error(contract, "success", "must be a boolean or None")
        if not self.terminated and not self.truncated and self.next_observation is None:
            raise validation_error(contract, "next_observation", "is required for a non-terminal step")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        object.__setattr__(self, "evaluation_signals", signals)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
