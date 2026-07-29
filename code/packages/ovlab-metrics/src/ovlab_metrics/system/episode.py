"""Episode extent and declared control-frequency metrics."""

from .._helpers import available, episode_result
from ..config import EmptyMetricConfig
from ..descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from ..plugin import EpisodeMetricPlugin
from ..requirements import MetricRequirements
from ..results import MetricStatus


class EpisodeLengthMetric(EpisodeMetricPlugin):
    descriptor = MetricDescriptor(
        "episode.length", "Episode length", "Number of accepted benchmark actions", "1.0.0",
        MetricLevel.SYSTEM, False, "control_step", OptimizationDirection.NONE,
        (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements()
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        return available(
            self, trace, config, len(trace.executed_actions), samples=len(trace.executed_actions),
            diagnostics={
                "executed_action_count": len(trace.executed_actions),
                "observation_count": len(trace.observations),
                "terminal_status": trace.terminal_status.value,
            },
        )


class ControlFrequencyMetric(EpisodeMetricPlugin):
    descriptor = MetricDescriptor(
        "system.control_frequency", "Control frequency", "Declared action-interface control frequency", "1.0.0",
        MetricLevel.SYSTEM, False, "Hz", OptimizationDirection.NONE,
        (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements()
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        if not trace.policy_predictions:
            return episode_result(
                self, trace, config, MetricStatus.UNAVAILABLE,
                reason="no action specification is stored",
            )
        frequencies = {
            prediction.action_spec.control_frequency_hz
            for prediction in trace.policy_predictions
        }
        if None in frequencies:
            return episode_result(
                self, trace, config, MetricStatus.UNAVAILABLE,
                reason="action specification has no declared control frequency",
            )
        if len(frequencies) != 1:
            return episode_result(
                self, trace, config, MetricStatus.ERROR,
                reason="action specifications disagree on control frequency",
            )
        frequency = frequencies.pop()
        return available(
            self, trace, config, float(frequency), samples=len(trace.executed_actions),
            diagnostics={"source": "action_spec.control_frequency_hz", "measured": False},
        )
