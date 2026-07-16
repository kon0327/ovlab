"""Immutable episode and task metric results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ovlab_core.contracts import EpisodeId, Metadata, RunId, TaskId, normalize_metadata
from ovlab_core.contracts.metadata import normalize_contract_value

from .descriptor import MetricScope
from .errors import MetricValidationError


class MetricStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric_id: str
    metric_version: str
    scope: MetricScope
    status: MetricStatus
    value: Any | None
    unit: str
    sample_count: int
    run_id: RunId
    task_id: TaskId
    episode_id: EpisodeId | None
    reason: str | None
    diagnostics: Metadata
    metric_config: Metadata
    metric_config_hash: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        if self.status is MetricStatus.AVAILABLE and self.value is None:
            raise MetricValidationError("available metric results require a value")
        if self.status is not MetricStatus.AVAILABLE and self.value is not None:
            raise MetricValidationError("unavailable metric results must use value=None")
        if self.sample_count < 0:
            raise MetricValidationError("sample_count must be non-negative")
        value = None if self.value is None else normalize_contract_value(self.value, type(self).__name__, "value")
        if isinstance(value, float) and not np.isfinite(value):
            raise MetricValidationError("available values must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "diagnostics", normalize_metadata(self.diagnostics, type(self).__name__, "diagnostics"))
        object.__setattr__(self, "metric_config", normalize_metadata(self.metric_config, type(self).__name__, "metric_config"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))


@dataclass(frozen=True, slots=True)
class MetricSummary:
    mean: float
    standard_deviation: float
    median: float
    minimum: float
    maximum: float
    valid_episode_count: int
    unavailable_episode_count: int
    excluded_episode_count: int

    @classmethod
    def from_values(cls, values, unavailable=0, excluded=0):
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.size == 0:
            raise MetricValidationError("a summary requires at least one value")
        return cls(
            float(np.mean(array)),
            float(np.std(array, ddof=0)),
            float(np.median(array)),
            float(np.min(array)),
            float(np.max(array)),
            int(array.size),
            unavailable,
            excluded,
        )

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "valid_episode_count": self.valid_episode_count,
            "unavailable_episode_count": self.unavailable_episode_count,
            "excluded_episode_count": self.excluded_episode_count,
        }
