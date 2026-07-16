"""Exact action metric formulas and sequence-boundary behavior."""

import pytest

from helpers.metric_traces import synthetic_trace
from ovlab_metrics import ActionSequenceMetricConfig, ActionSource, MetricEvaluator, MetricRegistry
from ovlab_metrics.action import ActionVarianceMetric, Smoothness1Metric, Smoothness2Metric
from ovlab_metrics.results import MetricStatus


def evaluate(plugin, trace, config=None):
    registry = MetricRegistry((plugin,))
    configs = None if config is None else {plugin.descriptor.metric_id: config}
    return MetricEvaluator(registry).evaluate(trace, configs)[0]


def test_known_variance_and_smoothness_values() -> None:
    trace = synthetic_trace()
    variance = evaluate(ActionVarianceMetric(), trace)
    smooth_1 = evaluate(Smoothness1Metric(), trace)
    smooth_2 = evaluate(Smoothness2Metric(), trace)
    assert variance.value == pytest.approx(7 / 9)
    assert variance.diagnostics["per_dimension_variance"] == pytest.approx((14 / 9, 0))
    assert smooth_1.value == pytest.approx(1.5)
    assert smooth_2.value == pytest.approx(1.0)


def test_constant_sequence_is_zero_and_short_sequences_are_insufficient() -> None:
    constant = synthetic_trace(actions=((1, 1), (1, 1), (1, 1)))
    assert evaluate(ActionVarianceMetric(), constant).value == 0
    assert evaluate(Smoothness1Metric(), constant).value == 0
    assert evaluate(Smoothness2Metric(), constant).value == 0
    short = synthetic_trace(actions=((0, 0),))
    assert evaluate(ActionVarianceMetric(), short).status is MetricStatus.INSUFFICIENT_DATA
    assert evaluate(Smoothness2Metric(), synthetic_trace(actions=((0, 0), (1, 0)))).status is MetricStatus.INSUFFICIENT_DATA


def test_requested_and_applied_sources_differ_and_chunks_do_not_leak() -> None:
    trace = synthetic_trace(
        actions=((0, 0), (0, 0), (0, 0)), requested=((0, 0), (1, 0), (3, 0)), horizon=3
    )
    applied = evaluate(ActionVarianceMetric(), trace)
    requested = evaluate(
        ActionVarianceMetric(), trace, ActionSequenceMetricConfig(ActionSource.REQUESTED)
    )
    assert applied.value == 0
    assert requested.value == pytest.approx(7 / 9)


def test_step_gap_is_explicitly_unavailable() -> None:
    result = evaluate(ActionVarianceMetric(), synthetic_trace(step_indices=(0, 2, 3)))
    assert result.status is MetricStatus.UNAVAILABLE
    assert "gap" in result.reason
