"""Typed immutable metric configurations and deterministic hashes."""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math

from .errors import MetricValidationError


class ActionSource(str, Enum):
    APPLIED = "applied"
    REQUESTED = "requested"


def canonical_config(config) -> dict:
    def normalize(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(asdict(config))


def config_hash(config) -> str:
    encoded = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EmptyMetricConfig:
    pass


@dataclass(frozen=True, slots=True)
class ActionSequenceMetricConfig:
    action_source: ActionSource = ActionSource.APPLIED

    def __post_init__(self):
        if not isinstance(self.action_source, ActionSource):
            raise MetricValidationError("action_source must be an ActionSource")


@dataclass(frozen=True, slots=True)
class ActionModificationMetricConfig:
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5

    def __post_init__(self):
        if not math.isfinite(self.absolute_tolerance) or not math.isfinite(self.relative_tolerance):
            raise MetricValidationError("action modification tolerances must be finite")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise MetricValidationError("action modification tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class RepeatedNoOpMetricConfig:
    action_source: ActionSource = ActionSource.APPLIED
    action_indices: tuple[int, ...] | None = None
    norm_threshold: float = 1e-4
    minimum_consecutive_run_length: int = 3

    def __post_init__(self):
        if not isinstance(self.action_source, ActionSource) or not math.isfinite(self.norm_threshold):
            raise MetricValidationError("invalid repeated no-op action source or threshold")
        if self.norm_threshold < 0 or self.minimum_consecutive_run_length < 2:
            raise MetricValidationError("invalid repeated no-op thresholds")
        if self.action_indices is not None:
            indices = tuple(self.action_indices)
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices) or len(indices) != len(set(indices)):
                raise MetricValidationError("action_indices must be unique and non-negative")
            object.__setattr__(self, "action_indices", indices)


@dataclass(frozen=True, slots=True)
class GripperFlickerMetricConfig:
    action_source: ActionSource = ActionSource.APPLIED
    activation_threshold: float = 0.5
    deadband: float = 0.1
    maximum_stable_interval: int = 2
    minimum_dwell_steps: int = 1

    def __post_init__(self):
        if not isinstance(self.action_source, ActionSource):
            raise MetricValidationError("action_source must be an ActionSource")
        if not math.isfinite(self.activation_threshold) or not math.isfinite(self.deadband):
            raise MetricValidationError("gripper thresholds must be finite")
        if self.activation_threshold < 0 or self.deadband < 0:
            raise MetricValidationError("gripper thresholds must be non-negative")
        if self.maximum_stable_interval < 1 or self.minimum_dwell_steps < 1:
            raise MetricValidationError("gripper intervals must be positive")


@dataclass(frozen=True, slots=True)
class SuccessRateMetricConfig:
    aborted_policy_attributable: bool = True


def config_from_canonical(metric_id: str, value: dict):
    """Reconstruct a typed configuration recorded in an immutable runner plan."""
    if not isinstance(metric_id, str) or not isinstance(value, dict):
        raise MetricValidationError("recorded metric configuration must be a mapping")
    if metric_id in {"action.variance", "action.smoothness_1", "action.smoothness_2"}:
        return ActionSequenceMetricConfig(ActionSource(value["action_source"]))
    if metric_id == "failure.action_modification_rate":
        return ActionModificationMetricConfig(value["absolute_tolerance"], value["relative_tolerance"])
    if metric_id == "failure.repeated_no_op_rate":
        indices = value.get("action_indices")
        return RepeatedNoOpMetricConfig(
            ActionSource(value["action_source"]),
            None if indices is None else tuple(indices),
            value["norm_threshold"],
            value["minimum_consecutive_run_length"],
        )
    if metric_id == "failure.gripper_flicker_rate":
        return GripperFlickerMetricConfig(
            ActionSource(value["action_source"]), value["activation_threshold"], value["deadband"],
            value["maximum_stable_interval"], value["minimum_dwell_steps"],
        )
    if metric_id == "task.success_rate":
        return SuccessRateMetricConfig(value.get("aborted_policy_attributable", True))
    if value:
        raise MetricValidationError(f"metric {metric_id!r} has an unexpected recorded configuration")
    return EmptyMetricConfig()
