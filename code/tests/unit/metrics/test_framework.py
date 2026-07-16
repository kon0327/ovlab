"""Metric descriptor, registry, requirements, and evaluator tests."""

from dataclasses import replace

import pytest

from helpers.metric_traces import synthetic_trace
from ovlab_metrics import (
    ActionSequenceMetricConfig,
    MetricDescriptor,
    MetricEvaluator,
    MetricLevel,
    MetricRegistry,
    MetricScope,
    OptimizationDirection,
    config_hash,
)
from ovlab_metrics.errors import MetricRegistryError, MetricValidationError
from ovlab_metrics.plugin import EpisodeMetricPlugin
from ovlab_metrics.results import MetricStatus


def descriptor(metric_id="test.metric", version="1.0.0"):
    return MetricDescriptor(
        metric_id, "Test", "Test metric", version, MetricLevel.ACTION, False, "unit",
        OptimizationDirection.NONE, (MetricScope.EPISODE,),
    )


def test_descriptor_validation_hashing_and_taxonomy() -> None:
    with pytest.raises(MetricValidationError):
        descriptor("")
    first = descriptor()
    assert first == descriptor() and hash(first) == hash(descriptor())
    assert tuple(MetricLevel) == (MetricLevel.TASK, MetricLevel.ACTION, MetricLevel.SYSTEM)


def test_registry_rejects_duplicates_and_conflicting_versions_and_orders() -> None:
    class Plugin(EpisodeMetricPlugin):
        descriptor = descriptor("z.metric")
        default_config = ActionSequenceMetricConfig()
        def evaluate(self, trace, config): ...

    class Other(Plugin):
        descriptor = descriptor("a.metric")

    registry = MetricRegistry((Plugin(), Other()))
    assert [item.metric_id for item in registry.descriptors()] == ["a.metric", "z.metric"]
    with pytest.raises(MetricRegistryError, match="duplicate"):
        registry.register(Plugin())
    class Conflict(Plugin):
        descriptor = descriptor("z.metric", "2.0.0")
    with pytest.raises(MetricRegistryError, match="conflicting"):
        registry.register(Conflict())


def test_configuration_hash_is_deterministic_and_sensitive() -> None:
    first = ActionSequenceMetricConfig()
    assert config_hash(first) == config_hash(ActionSequenceMetricConfig())
    assert config_hash(first) != config_hash(replace(first, action_source=first.action_source.REQUESTED))


def test_default_evaluator_is_deterministic_and_missing_is_not_zero() -> None:
    trace = synthetic_trace(include_success=False, collision_values=())
    evaluator = MetricEvaluator(MetricRegistry.default())
    first = evaluator.evaluate(trace)
    second = evaluator.evaluate(trace)
    assert first == second
    by_id = {result.metric_id: result for result in first}
    assert by_id["task.success"].status is MetricStatus.UNAVAILABLE
    assert by_id["task.success"].value is None
    assert by_id["failure.collision_rate"].status is MetricStatus.UNAVAILABLE


def test_non_strict_evaluator_converts_unexpected_error_and_strict_raises() -> None:
    class Broken(EpisodeMetricPlugin):
        descriptor = descriptor("test.broken")
        default_config = ActionSequenceMetricConfig()
        def evaluate(self, trace, config):
            raise RuntimeError("sensitive details")

    trace = synthetic_trace()
    registry = MetricRegistry((Broken(),))
    result = MetricEvaluator(registry).evaluate(trace)[0]
    assert result.status is MetricStatus.ERROR and "sensitive" not in result.reason
    with pytest.raises(RuntimeError):
        MetricEvaluator(registry, strict=True).evaluate(trace)
