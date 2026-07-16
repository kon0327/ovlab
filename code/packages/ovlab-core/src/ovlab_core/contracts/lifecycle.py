"""Immutable run, task, episode, and step lifecycle context."""

from dataclasses import dataclass, field

from .errors import validation_error
from .identifiers import EpisodeId, RunId, StepId, TaskId
from .instruction import Instruction
from .metadata import Metadata, normalize_metadata
from .time import validate_timestamp_ns
from .version import OVLAB_CONTRACT_VERSION


def _non_negative_int(value: int, contract: str, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise validation_error(contract, field_name, "must be a non-negative integer")


def _non_empty(value: str, contract: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(contract, field_name, "must not be empty or whitespace-only")


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: RunId
    created_wall_time_utc_ns: int
    experiment_name: str
    seed: int
    contract_version: str = OVLAB_CONTRACT_VERSION
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.run_id, RunId):
            raise validation_error(contract, "run_id", "must be a RunId")
        validate_timestamp_ns(self.created_wall_time_utc_ns, contract, "created_wall_time_utc_ns")
        _non_empty(self.experiment_name, contract, "experiment_name")
        _non_negative_int(self.seed, contract, "seed")
        if self.contract_version != OVLAB_CONTRACT_VERSION:
            raise validation_error(
                contract, "contract_version", f"must equal OVLAB_CONTRACT_VERSION ({OVLAB_CONTRACT_VERSION})"
            )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class TaskContext:
    run_id: RunId
    task_id: TaskId
    suite_name: str
    task_name: str
    task_index: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.run_id, RunId):
            raise validation_error(contract, "run_id", "must be a RunId")
        if not isinstance(self.task_id, TaskId):
            raise validation_error(contract, "task_id", "must be a TaskId")
        _non_empty(self.suite_name, contract, "suite_name")
        _non_empty(self.task_name, contract, "task_name")
        _non_negative_int(self.task_index, contract, "task_index")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    run_id: RunId
    task_id: TaskId
    episode_id: EpisodeId
    rollout_index: int
    seed: int
    initial_instruction: Instruction
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.run_id, RunId):
            raise validation_error(contract, "run_id", "must be a RunId")
        if not isinstance(self.task_id, TaskId):
            raise validation_error(contract, "task_id", "must be a TaskId")
        if not isinstance(self.episode_id, EpisodeId):
            raise validation_error(contract, "episode_id", "must be an EpisodeId")
        _non_negative_int(self.rollout_index, contract, "rollout_index")
        _non_negative_int(self.seed, contract, "seed")
        if not isinstance(self.initial_instruction, Instruction):
            raise validation_error(contract, "initial_instruction", "must be an Instruction")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class StepContext:
    run_id: RunId
    task_id: TaskId
    episode_id: EpisodeId
    step_id: StepId
    step_index: int
    timestamp_ns: int

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.run_id, RunId):
            raise validation_error(contract, "run_id", "must be a RunId")
        if not isinstance(self.task_id, TaskId):
            raise validation_error(contract, "task_id", "must be a TaskId")
        if not isinstance(self.episode_id, EpisodeId):
            raise validation_error(contract, "episode_id", "must be an EpisodeId")
        if not isinstance(self.step_id, StepId):
            raise validation_error(contract, "step_id", "must be a StepId")
        _non_negative_int(self.step_index, contract, "step_index")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
