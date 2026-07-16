"""Inspectable trace requirements and deterministic resolution."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ovlab_core.contracts import EpisodeTrace, SignalAccess

from .errors import MetricValidationError


class TraceField(str, Enum):
    EXECUTED_ACTIONS = "executed_actions"
    ACTION_SPECIFICATION = "action_specification"
    POLICY_PREDICTIONS = "policy_predictions"
    PREDICTION_VALIDITY = "prediction_validity"
    INFERENCE_DURATION = "inference_duration"
    TERMINAL_STATUS = "terminal_status"
    STEP_TIMESTAMPS = "step_timestamps"


@dataclass(frozen=True, slots=True)
class TraceFieldRequirement:
    field: TraceField
    required: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class SignalRequirement:
    name: str
    expected_dtype: str
    expected_shape: tuple[int, ...]
    allowed_access: tuple[SignalAccess, ...]
    required: bool = True
    description: str = ""

    def __post_init__(self):
        if not self.name.strip() or not self.allowed_access:
            raise MetricValidationError("signal requirements need a name and allowed access classes")
        object.__setattr__(self, "expected_shape", tuple(self.expected_shape))
        object.__setattr__(self, "allowed_access", tuple(self.allowed_access))


@dataclass(frozen=True, slots=True)
class MinimumSampleRequirement:
    source: TraceField
    minimum: int
    description: str = ""

    def __post_init__(self):
        if self.minimum <= 0:
            raise MetricValidationError("minimum samples must be positive")


@dataclass(frozen=True, slots=True)
class MetricRequirements:
    fields: tuple[TraceFieldRequirement, ...] = ()
    signals: tuple[SignalRequirement, ...] = ()
    minimum_samples: tuple[MinimumSampleRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class RequirementResolution:
    available: bool
    missing_requirements: tuple[str, ...] = ()
    incompatible_requirements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    insufficient_requirements: tuple[str, ...] = ()


def _field_count(trace: EpisodeTrace, field: TraceField) -> int:
    if field in (TraceField.EXECUTED_ACTIONS, TraceField.ACTION_SPECIFICATION):
        return len(trace.executed_actions)
    if field in (TraceField.POLICY_PREDICTIONS, TraceField.PREDICTION_VALIDITY, TraceField.INFERENCE_DURATION):
        return len(trace.policy_predictions)
    if field is TraceField.STEP_TIMESTAMPS:
        return len(trace.step_contexts)
    return 1


def resolve_requirements(trace: EpisodeTrace, requirements: MetricRequirements) -> RequirementResolution:
    missing, incompatible, warnings, insufficient = [], [], [], []
    for requirement in requirements.fields:
        count = _field_count(trace, requirement.field)
        if count == 0 and requirement.required:
            missing.append(requirement.field.value)
    signals_by_name = {}
    for signal in trace.evaluation_signals:
        signals_by_name.setdefault(signal.name, []).append(signal)
    for requirement in requirements.signals:
        values = signals_by_name.get(requirement.name, [])
        if not values:
            if requirement.required:
                missing.append(f"signal:{requirement.name}")
            else:
                warnings.append(f"optional signal {requirement.name} is absent")
            continue
        for signal in values:
            array = np.asarray(signal.value)
            if array.shape != requirement.expected_shape or array.dtype != np.dtype(requirement.expected_dtype):
                incompatible.append(f"signal:{requirement.name}:shape_or_dtype")
                break
            if signal.access is None:
                warnings.append(f"signal {requirement.name} access class is unknown in this trace")
            elif signal.access not in requirement.allowed_access:
                incompatible.append(f"signal:{requirement.name}:access")
                break
    for requirement in requirements.minimum_samples:
        count = _field_count(trace, requirement.source)
        if count < requirement.minimum:
            insufficient.append(f"{requirement.source.value}:{count}<{requirement.minimum}")
    return RequirementResolution(
        not missing and not incompatible and not insufficient,
        tuple(sorted(set(missing))),
        tuple(sorted(set(incompatible))),
        tuple(sorted(set(warnings))),
        tuple(sorted(set(insufficient))),
    )
