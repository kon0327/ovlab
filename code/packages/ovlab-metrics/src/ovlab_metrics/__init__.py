"""Stable public API for offline OVLAB metric evaluation."""

from .aggregation import aggregate_episode_results, aggregate_results_by_task
from .config import (
    ActionModificationMetricConfig,
    ActionSequenceMetricConfig,
    ActionSource,
    EmptyMetricConfig,
    GripperFlickerMetricConfig,
    RepeatedNoOpMetricConfig,
    SuccessRateMetricConfig,
    canonical_config,
    config_hash,
)
from .descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from .evaluator import MetricEvaluator
from .plugin import EpisodeMetricPlugin, TaskMetricPlugin
from .registry import MetricRegistry
from .requirements import (
    MetricRequirements,
    MinimumSampleRequirement,
    RequirementResolution,
    SignalRequirement,
    TraceField,
    TraceFieldRequirement,
    resolve_requirements,
)
from .results import MetricResult, MetricStatus, MetricSummary

__all__ = [
    "ActionModificationMetricConfig", "ActionSequenceMetricConfig", "ActionSource",
    "EmptyMetricConfig", "EpisodeMetricPlugin", "GripperFlickerMetricConfig",
    "MetricDescriptor", "MetricEvaluator", "MetricLevel", "MetricRegistry",
    "MetricRequirements", "MetricResult", "MetricScope", "MetricStatus", "MetricSummary",
    "MinimumSampleRequirement", "OptimizationDirection", "RepeatedNoOpMetricConfig",
    "RequirementResolution", "SignalRequirement", "SuccessRateMetricConfig", "TaskMetricPlugin",
    "TraceField", "TraceFieldRequirement", "aggregate_episode_results", "aggregate_results_by_task", "canonical_config",
    "config_hash", "resolve_requirements",
]
