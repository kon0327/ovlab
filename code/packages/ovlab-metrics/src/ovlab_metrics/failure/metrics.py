"""Cross-cutting failure-indicator metrics."""

from collections import Counter

import numpy as np

from ovlab_core.contracts import GripperConvention, PredictionValidity, SignalAccess

from .._helpers import available, episode_result
from ..action_sequence import action_spec_identity, extract_action_sequence
from ..config import (
    ActionModificationMetricConfig,
    EmptyMetricConfig,
    GripperFlickerMetricConfig,
    RepeatedNoOpMetricConfig,
)
from ..descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from ..plugin import EpisodeMetricPlugin
from ..requirements import MetricRequirements, SignalRequirement, TraceField, TraceFieldRequirement
from ..results import MetricStatus


def failure_descriptor(metric_id, name, description, level=MetricLevel.ACTION):
    return MetricDescriptor(
        metric_id, name, description, "1.0.0", level, True, "ratio",
        OptimizationDirection.LOWER, (MetricScope.EPISODE, MetricScope.TASK),
    )


class InvalidPredictionRateMetric(EpisodeMetricPlugin):
    descriptor = failure_descriptor(
        "failure.invalid_prediction_rate", "Invalid prediction rate", "Non-VALID predictions per stored prediction",
        MetricLevel.SYSTEM,
    )
    requirements = MetricRequirements((TraceFieldRequirement(TraceField.PREDICTION_VALIDITY),))
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        if not trace.policy_predictions:
            return episode_result(self, trace, config, MetricStatus.UNAVAILABLE, reason="no predictions are stored")
        invalid = sum(prediction.validity is not PredictionValidity.VALID for prediction in trace.policy_predictions)
        return available(
            self, trace, config, invalid / len(trace.policy_predictions), samples=len(trace.policy_predictions),
            diagnostics={"invalid_count": invalid, "prediction_count": len(trace.policy_predictions)},
        )


class ActionModificationRateMetric(EpisodeMetricPlugin):
    descriptor = failure_descriptor(
        "failure.action_modification_rate", "Action modification rate", "Requested commands changed before execution"
    )
    requirements = MetricRequirements((TraceFieldRequirement(TraceField.EXECUTED_ACTIONS),))
    default_config = ActionModificationMetricConfig()

    def evaluate(self, trace, config):
        actions = trace.executed_actions
        differences = np.asarray(
            [np.linalg.norm(action.applied_action - action.requested_action) for action in actions], dtype=np.float64
        )
        modified = [
            not np.allclose(action.requested_action, action.applied_action, atol=config.absolute_tolerance, rtol=config.relative_tolerance)
            for action in actions
        ]
        reasons = Counter(
            action.modification_reason for action, changed in zip(actions, modified) if changed and action.modification_reason
        )
        return available(
            self, trace, config, sum(modified) / len(actions), samples=len(actions),
            diagnostics={
                "modification_count": sum(modified),
                "maximum_modification_norm": float(np.max(differences)),
                "mean_modification_norm": float(np.mean(differences)),
                "modification_reasons": dict(sorted(reasons.items())),
            },
        )


class RepeatedNoOpRateMetric(EpisodeMetricPlugin):
    descriptor = failure_descriptor(
        "failure.repeated_no_op_rate", "Repeated no-op rate", "Steps belonging to sufficiently long no-op runs"
    )
    requirements = MetricRequirements((TraceFieldRequirement(TraceField.EXECUTED_ACTIONS),))
    default_config = RepeatedNoOpMetricConfig()

    def evaluate(self, trace, config):
        sequence = extract_action_sequence(trace, config.action_source)
        indices = config.action_indices
        if indices is None:
            gripper = set(sequence.action_spec.gripper_indices)
            indices = tuple(index for index in range(sequence.action_spec.dimension) if index not in gripper)
        if not indices or any(index >= sequence.action_spec.dimension for index in indices):
            return episode_result(self, trace, config, MetricStatus.NOT_APPLICABLE, reason="no eligible action indices")
        flags = np.linalg.norm(sequence.values[:, indices], axis=1) <= config.norm_threshold
        runs, start = [], None
        for index, flag in enumerate(tuple(flags) + (False,)):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                length = index - start
                if length >= config.minimum_consecutive_run_length:
                    runs.append(length)
                start = None
        repeated_steps = sum(runs)
        return available(
            self, trace, config, repeated_steps / len(flags), samples=len(flags),
            diagnostics={"run_count": len(runs), "maximum_run_length": max(runs, default=0), "repeated_steps": repeated_steps},
            metadata={"action_source": sequence.source.value, "action_spec": action_spec_identity(sequence.action_spec)},
        )


class GripperFlickerRateMetric(EpisodeMetricPlugin):
    descriptor = failure_descriptor(
        "failure.gripper_flicker_rate", "Gripper flicker rate", "Rapid alternating gripper transitions"
    )
    requirements = MetricRequirements((TraceFieldRequirement(TraceField.EXECUTED_ACTIONS),))
    default_config = GripperFlickerMetricConfig()

    def evaluate(self, trace, config):
        sequence = extract_action_sequence(trace, config.action_source)
        spec = sequence.action_spec
        if not spec.gripper_indices:
            return episode_result(self, trace, config, MetricStatus.NOT_APPLICABLE, reason="action has no gripper")
        if spec.gripper_convention is GripperConvention.NONE:
            return episode_result(self, trace, config, MetricStatus.UNAVAILABLE, reason="gripper convention unavailable")
        values = np.mean(sequence.values[:, spec.gripper_indices], axis=1)
        states = []
        positive_closed = spec.gripper_convention in (
            GripperConvention.CLOSED_POSITIVE,
            GripperConvention.BINARY_CLOSED_ONE,
        )
        binary = spec.gripper_convention in (
            GripperConvention.BINARY_OPEN_ONE,
            GripperConvention.BINARY_CLOSED_ONE,
        )
        for index, value in enumerate(values):
            if binary and value >= config.activation_threshold + config.deadband:
                state = "closed" if positive_closed else "open"
            elif binary and value <= config.activation_threshold - config.deadband:
                state = "open" if positive_closed else "closed"
            elif not binary and value >= config.activation_threshold:
                state = "closed" if positive_closed else "open"
            elif not binary and value <= -config.activation_threshold:
                state = "open" if positive_closed else "closed"
            else:
                continue
            states.append((index, state))
        transitions = [(index, state) for (previous_index, previous), (index, state) in zip(states, states[1:]) if state != previous]
        flickers = 0
        for first, second in zip(transitions, transitions[1:]):
            if first[1] != second[1] and second[0] - first[0] <= config.maximum_stable_interval:
                flickers += 1
        denominator = max(len(states) - 1, 0)
        if denominator == 0:
            return available(self, trace, config, 0.0, samples=len(states), diagnostics={"transitions": 0, "flickers": 0})
        return available(
            self, trace, config, flickers / denominator, samples=len(states),
            diagnostics={"transitions": len(transitions), "flickers": flickers, "unknown_steps": len(values) - len(states)},
            metadata={"action_source": sequence.source.value, "action_spec": action_spec_identity(spec)},
        )


class CollisionRateMetric(EpisodeMetricPlugin):
    descriptor = failure_descriptor(
        "failure.collision_rate", "Collision rate", "Explicit semantic collision events per sample"
    )
    requirements = MetricRequirements(
        signals=(SignalRequirement(
            "safety.collision_event", "bool", (), (SignalAccess.EVALUATION_ONLY, SignalAccess.PRIVILEGED)
        ),)
    )
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        signals = [signal for signal in trace.evaluation_signals if signal.name == "safety.collision_event"]
        if not signals:
            return episode_result(self, trace, config, MetricStatus.UNAVAILABLE, reason="semantic collision signal missing")
        collisions = sum(bool(signal.value) for signal in signals)
        return available(
            self, trace, config, collisions / len(signals), samples=len(signals),
            diagnostics={"collision_count": collisions},
        )
