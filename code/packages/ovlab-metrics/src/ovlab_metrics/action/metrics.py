"""Episode action variance and discrete smoothness metrics."""

import numpy as np

from .._helpers import available
from ..action_sequence import action_spec_identity, extract_action_sequence
from ..config import ActionSequenceMetricConfig
from ..descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from ..plugin import EpisodeMetricPlugin
from ..requirements import MetricRequirements, MinimumSampleRequirement, TraceField, TraceFieldRequirement


class _ActionMetric(EpisodeMetricPlugin):
    default_config = ActionSequenceMetricConfig()

    def sequence(self, trace, config):
        return extract_action_sequence(trace, config.action_source)

    def metadata(self, sequence):
        return {"action_source": sequence.source.value, "action_spec": action_spec_identity(sequence.action_spec)}


class ActionVarianceMetric(_ActionMetric):
    descriptor = MetricDescriptor(
        "action.variance", "Action variance", "Mean population variance across action dimensions", "1.0.0",
        MetricLevel.ACTION, False, "action_unit_squared", OptimizationDirection.NONE, (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements(
        (TraceFieldRequirement(TraceField.EXECUTED_ACTIONS), TraceFieldRequirement(TraceField.ACTION_SPECIFICATION)),
        minimum_samples=(MinimumSampleRequirement(TraceField.EXECUTED_ACTIONS, 2),),
    )

    def evaluate(self, trace, config):
        sequence = self.sequence(trace, config)
        per_dimension = np.var(sequence.values, axis=0, ddof=0)
        return available(
            self, trace, config, float(np.mean(per_dimension)), samples=len(sequence.values),
            diagnostics={"per_dimension_variance": per_dimension.tolist()}, metadata=self.metadata(sequence),
        )


class Smoothness1Metric(_ActionMetric):
    descriptor = MetricDescriptor(
        "action.smoothness_1", "Smooth_1", "Mean first command difference norm", "1.0.0",
        MetricLevel.ACTION, False, "action_unit/control_step", OptimizationDirection.LOWER, (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements(
        (TraceFieldRequirement(TraceField.EXECUTED_ACTIONS),),
        minimum_samples=(MinimumSampleRequirement(TraceField.EXECUTED_ACTIONS, 2),),
    )

    def evaluate(self, trace, config):
        sequence = self.sequence(trace, config)
        value = float(np.mean(np.linalg.norm(np.diff(sequence.values, axis=0), axis=1)))
        return available(self, trace, config, value, samples=len(sequence.values), metadata=self.metadata(sequence))


class Smoothness2Metric(_ActionMetric):
    descriptor = MetricDescriptor(
        "action.smoothness_2", "Smooth_2", "Mean second command difference norm", "1.0.0",
        MetricLevel.ACTION, False, "action_unit/control_step^2", OptimizationDirection.LOWER, (MetricScope.EPISODE, MetricScope.TASK),
    )
    requirements = MetricRequirements(
        (TraceFieldRequirement(TraceField.EXECUTED_ACTIONS),),
        minimum_samples=(MinimumSampleRequirement(TraceField.EXECUTED_ACTIONS, 3),),
    )

    def evaluate(self, trace, config):
        sequence = self.sequence(trace, config)
        second = sequence.values[2:] - 2 * sequence.values[1:-1] + sequence.values[:-2]
        value = float(np.mean(np.linalg.norm(second, axis=1)))
        return available(self, trace, config, value, samples=len(sequence.values), metadata=self.metadata(sequence))
