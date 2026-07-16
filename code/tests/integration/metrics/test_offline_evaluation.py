"""Recompute the default metric set repeatedly from immutable traces."""

from helpers.metric_traces import synthetic_trace
from ovlab_metrics import MetricEvaluator, MetricLevel, MetricRegistry


def test_default_metrics_recompute_deterministically_offline() -> None:
    trace = synthetic_trace(collision_values=(False, True))
    before = trace.executed_actions[0].applied_action.tobytes()
    registry = MetricRegistry.default()
    evaluator = MetricEvaluator(registry)
    first = evaluator.evaluate(trace)
    second = evaluator.evaluate(trace)
    assert first == second
    assert trace.executed_actions[0].applied_action.tobytes() == before
    levels = {descriptor.metric_level for descriptor in registry.descriptors()}
    assert levels == {MetricLevel.TASK, MetricLevel.ACTION, MetricLevel.SYSTEM}
    assert any(descriptor.is_failure_indicator for descriptor in registry.descriptors())
    assert "failure" not in {level.value for level in MetricLevel}
