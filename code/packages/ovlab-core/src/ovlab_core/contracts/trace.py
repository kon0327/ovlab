"""Immutable in-memory episode traces for later metric recomputation."""

from dataclasses import dataclass, field
from enum import Enum

from .actions import ActionPrediction, ExecutedAction
from .errors import validation_error
from .instruction import Instruction
from .lifecycle import EpisodeContext, StepContext
from .metadata import Metadata, normalize_metadata
from .observation import PolicyObservation
from .signals import SignalValue
from .time import validate_optional_timestamp_ns, validate_timestamp_ns


class EpisodeTerminalStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIME_LIMIT = "time_limit"
    POLICY_ERROR = "policy_error"
    BENCHMARK_ERROR = "benchmark_error"
    ABORTED = "aborted"


def _validate_order(events: tuple, contract: str, field_name: str) -> None:
    timestamps = [event.timestamp_ns for event in events]
    if timestamps != sorted(timestamps):
        raise validation_error(contract, field_name, "events must be ordered by timestamp_ns")


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Raw immutable episode evidence; metric results are intentionally excluded."""

    episode_context: EpisodeContext
    step_contexts: tuple[StepContext, ...]
    observations: tuple[PolicyObservation, ...]
    instruction_events: tuple[Instruction, ...]
    policy_predictions: tuple[ActionPrediction, ...]
    executed_actions: tuple[ExecutedAction, ...]
    evaluation_signals: tuple[SignalValue, ...]
    terminal_status: EpisodeTerminalStatus
    start_timestamp_ns: int
    end_timestamp_ns: int | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.episode_context, EpisodeContext):
            raise validation_error(contract, "episode_context", "must be an EpisodeContext")
        if not isinstance(self.terminal_status, EpisodeTerminalStatus):
            raise validation_error(contract, "terminal_status", "must be an EpisodeTerminalStatus")
        validate_timestamp_ns(self.start_timestamp_ns, contract, "start_timestamp_ns")
        validate_optional_timestamp_ns(self.end_timestamp_ns, contract, "end_timestamp_ns")
        if self.end_timestamp_ns is not None and self.start_timestamp_ns > self.end_timestamp_ns:
            raise validation_error(contract, "end_timestamp_ns", "must be greater than or equal to start_timestamp_ns")

        typed_collections = (
            ("step_contexts", StepContext),
            ("observations", PolicyObservation),
            ("instruction_events", Instruction),
            ("policy_predictions", ActionPrediction),
            ("executed_actions", ExecutedAction),
            ("evaluation_signals", SignalValue),
        )
        collections = {}
        for field_name, expected_type in typed_collections:
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, expected_type) for value in values):
                raise validation_error(contract, field_name, f"must contain only {expected_type.__name__} values")
            collections[field_name] = values
            object.__setattr__(self, field_name, values)

        contexts = collections["step_contexts"]
        expected_ids = (
            self.episode_context.run_id,
            self.episode_context.task_id,
            self.episode_context.episode_id,
        )
        seen_step_ids = set()
        for context in contexts:
            if (context.run_id, context.task_id, context.episode_id) != expected_ids:
                raise validation_error(contract, "step_contexts", "contains a step from a different run/task/episode")
            if context.step_id in seen_step_ids:
                raise validation_error(contract, "step_contexts", "contains a duplicate step_id")
            seen_step_ids.add(context.step_id)
        if [context.step_index for context in contexts] != sorted(context.step_index for context in contexts):
            raise validation_error(contract, "step_contexts", "must be ordered by step_index")

        for field_name in ("observations", "policy_predictions", "executed_actions"):
            for event in collections[field_name]:
                if event.step_id not in seen_step_ids:
                    raise validation_error(contract, field_name, "contains an event outside this episode's steps")
        for signal in collections["evaluation_signals"]:
            if signal.step_id is not None and signal.step_id not in seen_step_ids:
                raise validation_error(contract, "evaluation_signals", "contains an event outside this episode's steps")

        for field_name in (
            "step_contexts",
            "observations",
            "instruction_events",
            "policy_predictions",
            "executed_actions",
            "evaluation_signals",
        ):
            _validate_order(collections[field_name], contract, field_name)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
