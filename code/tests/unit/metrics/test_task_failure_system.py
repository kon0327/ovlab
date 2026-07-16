"""Task, failure-indicator, and system metric tests."""

import pytest

from helpers.metric_traces import metric_action_spec, synthetic_trace
from ovlab_core.contracts import EpisodeTerminalStatus, GripperConvention, PredictionValidity
from ovlab_core.contracts import TaskContext
from ovlab_metrics import MetricEvaluator, MetricRegistry
from ovlab_metrics.failure import (
    ActionModificationRateMetric,
    CollisionRateMetric,
    GripperFlickerRateMetric,
    InvalidPredictionRateMetric,
    RepeatedNoOpRateMetric,
)
from ovlab_metrics.results import MetricStatus
from ovlab_metrics.system import InferenceLatencyMetric
from ovlab_metrics.task import TaskSuccessMetric, TaskSuccessRateMetric


def evaluate(plugin, trace):
    return MetricEvaluator(MetricRegistry((plugin,))).evaluate(trace)[0]


def test_success_failure_missing_and_contradiction() -> None:
    assert evaluate(TaskSuccessMetric(), synthetic_trace()).value == 1
    failure = synthetic_trace(terminal_status=EpisodeTerminalStatus.FAILURE, success_signal=False)
    assert evaluate(TaskSuccessMetric(), failure).value == 0
    missing = synthetic_trace(include_success=False)
    assert evaluate(TaskSuccessMetric(), missing).status is MetricStatus.UNAVAILABLE
    contradiction = synthetic_trace(terminal_status=EpisodeTerminalStatus.FAILURE, success_signal=True)
    assert evaluate(TaskSuccessMetric(), contradiction).status is MetricStatus.ERROR


def test_success_rate_denominator_policy() -> None:
    traces = (
        synthetic_trace(terminal_status=EpisodeTerminalStatus.SUCCESS, success_signal=True),
        synthetic_trace(terminal_status=EpisodeTerminalStatus.FAILURE, success_signal=False),
        synthetic_trace(terminal_status=EpisodeTerminalStatus.TIME_LIMIT, success_signal=False),
        synthetic_trace(terminal_status=EpisodeTerminalStatus.POLICY_ERROR, success_signal=False),
        synthetic_trace(terminal_status=EpisodeTerminalStatus.BENCHMARK_ERROR, success_signal=False),
    )
    episode_results = tuple(evaluate(TaskSuccessMetric(), trace) for trace in traces)
    context = traces[0].episode_context
    task = TaskContext(context.run_id, context.task_id, "synthetic", "task", 0)
    result = TaskSuccessRateMetric().aggregate(task, episode_results, TaskSuccessRateMetric.default_config)
    assert result.value == pytest.approx(1 / 4)
    assert result.diagnostics["numerator"] == 1
    assert result.diagnostics["denominator"] == 4
    assert result.diagnostics["excluded_episode_count"] == 1


def test_invalid_prediction_rate_and_missing_predictions() -> None:
    trace = synthetic_trace(
        prediction_validities=(PredictionValidity.VALID, PredictionValidity.MODEL_ERROR, PredictionValidity.INVALID_VALUE)
    )
    result = evaluate(InvalidPredictionRateMetric(), trace)
    assert result.value == pytest.approx(2 / 3)
    empty = synthetic_trace(actions=(), include_success=False, inference_durations=())
    assert evaluate(InvalidPredictionRateMetric(), empty).status is MetricStatus.UNAVAILABLE


def test_action_modification_and_repeated_noop() -> None:
    modified = synthetic_trace(
        actions=((0, 0), (0.2, 0), (0, 0)), requested=((0, 0), (0, 0), (0, 0))
    )
    result = evaluate(ActionModificationRateMetric(), modified)
    assert result.value == pytest.approx(1 / 3)
    noops = evaluate(RepeatedNoOpRateMetric(), synthetic_trace(actions=((0, 0),) * 4))
    assert noops.value == 1
    assert noops.diagnostics["maximum_run_length"] == 4


@pytest.mark.parametrize(
    "convention,actions",
    [
        (GripperConvention.CLOSED_POSITIVE, ((0, -1), (0, 1), (0, -1))),
        (GripperConvention.OPEN_POSITIVE, ((0, 1), (0, -1), (0, 1))),
        (GripperConvention.BINARY_CLOSED_ONE, ((0, 0), (0, 1), (0, 0))),
        (GripperConvention.BINARY_OPEN_ONE, ((0, 1), (0, 0), (0, 1))),
    ],
)
def test_gripper_flicker_supported_conventions(convention, actions) -> None:
    spec = metric_action_spec(2, gripper=True, convention=convention)
    result = evaluate(GripperFlickerRateMetric(), synthetic_trace(actions=actions, action_spec=spec))
    assert result.status is MetricStatus.AVAILABLE
    assert result.diagnostics["flickers"] >= 1


def test_no_gripper_is_not_applicable() -> None:
    result = evaluate(GripperFlickerRateMetric(), synthetic_trace())
    assert result.status is MetricStatus.NOT_APPLICABLE


def test_collision_requires_explicit_semantic_signal() -> None:
    assert evaluate(CollisionRateMetric(), synthetic_trace()).status is MetricStatus.UNAVAILABLE
    result = evaluate(CollisionRateMetric(), synthetic_trace(collision_values=(False, True, True)))
    assert result.value == pytest.approx(2 / 3)


def test_inference_latency_summary_and_linear_p95() -> None:
    result = evaluate(InferenceLatencyMetric(), synthetic_trace(inference_durations=(1_000_000, 2_000_000, 4_000_000)))
    assert result.value == pytest.approx(7 / 3)
    assert result.diagnostics["p95"] == pytest.approx(3.8)
    assert "rpc" not in result.diagnostics
