"""Metric framework errors."""


class MetricError(Exception):
    """Base metric error."""


class MetricValidationError(MetricError, ValueError):
    """A metric contract or configuration is invalid."""


class MetricRegistryError(MetricError):
    """Metric registration or resolution failed."""


class MetricAggregationError(MetricError):
    """Episode results cannot be aggregated safely."""


class ActionSequenceError(MetricError):
    """Executed actions do not form one valid contiguous sequence."""
