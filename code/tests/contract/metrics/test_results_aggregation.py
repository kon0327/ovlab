"""Immutable result and safe macro aggregation contracts."""

from dataclasses import replace

import pytest

from helpers.contexts import make_run_context
from helpers.metric_traces import synthetic_trace
from ovlab_core.contracts import TaskContext
from ovlab_metrics import MetricEvaluator, MetricRegistry, aggregate_episode_results
from ovlab_metrics.action import Smoothness1Metric
from ovlab_metrics.errors import MetricAggregationError


def task_context(trace):
    episode = trace.episode_context
    return TaskContext(episode.run_id, episode.task_id, "synthetic", "task", 0)


def test_macro_aggregation_and_population_standard_deviation() -> None:
    evaluator = MetricEvaluator(MetricRegistry((Smoothness1Metric(),)))
    first_trace = synthetic_trace(actions=((0, 0), (1, 0), (2, 0)))
    second_trace = synthetic_trace(actions=((0, 0), (3, 0), (6, 0)))
    first = evaluator.evaluate(first_trace)[0]
    second = evaluator.evaluate(second_trace)[0]
    summary = aggregate_episode_results(task_context(first_trace), (first, second))
    assert summary.value["mean"] == 2
    assert summary.value["standard_deviation"] == 1
    assert summary.value["valid_episode_count"] == 2


def test_aggregation_rejects_mixed_configuration_and_errors() -> None:
    trace = synthetic_trace()
    result = MetricEvaluator(MetricRegistry((Smoothness1Metric(),))).evaluate(trace)[0]
    with pytest.raises(MetricAggregationError):
        aggregate_episode_results(task_context(trace), (result, replace(result, metric_config_hash="different")))
