"""Internal policy inference-duration metric."""

import numpy as np

from .._helpers import available, episode_result
from ..config import EmptyMetricConfig
from ..descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from ..plugin import EpisodeMetricPlugin
from ..requirements import MetricRequirements, TraceField, TraceFieldRequirement
from ..results import MetricStatus


class InferenceLatencyMetric(EpisodeMetricPlugin):
    descriptor = MetricDescriptor(
        "system.inference_latency", "Inference latency", "Stored internal policy inference duration", "1.0.0",
        MetricLevel.SYSTEM, False, "ms", OptimizationDirection.LOWER, (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements((TraceFieldRequirement(TraceField.INFERENCE_DURATION),))
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        if not trace.policy_predictions:
            return episode_result(self, trace, config, MetricStatus.UNAVAILABLE, reason="no inference samples stored")
        nanoseconds = np.asarray([prediction.inference_duration_ns for prediction in trace.policy_predictions], dtype=np.int64)
        milliseconds = nanoseconds.astype(np.float64) / 1_000_000.0
        summary = {
            "mean": float(np.mean(milliseconds)),
            "median": float(np.median(milliseconds)),
            "p95": float(np.percentile(milliseconds, 95, method="linear")),
            "minimum": float(np.min(milliseconds)),
            "maximum": float(np.max(milliseconds)),
            "standard_deviation": float(np.std(milliseconds, ddof=0)),
        }
        return available(self, trace, config, summary["mean"], samples=len(milliseconds), diagnostics=summary)
