"""Validated immutable lifecycle identifiers."""

from dataclasses import dataclass

from .errors import validation_error


@dataclass(frozen=True, slots=True)
class _StringIdentifier:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise validation_error(type(self).__name__, "value", "must be a string")
        if not self.value.strip():
            raise validation_error(type(self).__name__, "value", "must not be empty or whitespace-only")

    def __str__(self) -> str:
        return self.value


class RunId(_StringIdentifier):
    """Externally assigned run identifier."""


class TaskId(_StringIdentifier):
    """Externally assigned task identifier."""


class EpisodeId(_StringIdentifier):
    """Externally assigned episode identifier."""


class StepId(_StringIdentifier):
    """Externally assigned step identifier."""


class InstructionId(_StringIdentifier):
    """Externally assigned instruction-event identifier."""


class PredictionId(_StringIdentifier):
    """Externally assigned policy-prediction identifier."""
