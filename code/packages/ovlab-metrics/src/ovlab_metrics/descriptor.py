"""Stable metric identity and taxonomy."""

from dataclasses import dataclass, field
from enum import Enum
import json

from ovlab_core.contracts import Metadata, normalize_metadata

from .errors import MetricValidationError


class MetricLevel(str, Enum):
    TASK = "task"
    ACTION = "action"
    SYSTEM = "system"


class MetricScope(str, Enum):
    EPISODE = "episode"
    TASK = "task"
    RUN = "run"


class OptimizationDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MetricDescriptor:
    metric_id: str
    display_name: str
    description: str
    metric_version: str
    metric_level: MetricLevel
    is_failure_indicator: bool
    result_unit: str
    optimization_direction: OptimizationDirection
    supported_scopes: tuple[MetricScope, ...]
    metadata: Metadata = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        for name in ("metric_id", "display_name", "description", "metric_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MetricValidationError(f"{name} must be non-empty")
        if not self.metric_id.replace(".", "_").replace("-", "_").replace("_", "a").isalnum():
            raise MetricValidationError("metric_id must be machine-readable")
        scopes = tuple(self.supported_scopes)
        if not scopes or any(not isinstance(scope, MetricScope) for scope in scopes):
            raise MetricValidationError("supported_scopes must contain MetricScope values")
        object.__setattr__(self, "supported_scopes", tuple(sorted(set(scopes), key=lambda value: value.value)))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))

    def __hash__(self) -> int:
        def thaw(value):
            if hasattr(value, "items"):
                return {key: thaw(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value

        metadata = json.dumps(thaw(self.metadata), sort_keys=True, separators=(",", ":"))
        return hash((self.metric_id, self.metric_version, self.metric_level, self.is_failure_indicator, metadata))
