"""Independent fixed-value oracle for Gate H.1 qualification metrics."""

import math

import numpy as np
import pytest

from helpers.metric_traces import metric_action_spec, synthetic_trace
from ovlab_core.contracts import (
    ActionPrediction, ActionRepresentation, ActionSpec, EpisodeTerminalStatus,
    GripperConvention, PredictionId, PredictionValidity, RotationRepresentation,
)
from ovlab_metrics import MetricEvaluator, MetricRegistry, MetricStatus
from ovlab_metrics.action.metrics import ActionVarianceMetric, Smoothness1Metric, Smoothness2Metric
from ovlab_metrics.failure.metrics import GripperFlickerRateMetric, InvalidPredictionRateMetric
from ovlab_metrics.system.episode import ControlFrequencyMetric, EpisodeLengthMetric
from ovlab_metrics.system.inference_latency import InferenceLatencyMetric
from ovlab_metrics.task.success import TaskSuccessMetric


def _evaluate(trace, *plugins):
    return {
        result.metric_id: result
        for result in MetricEvaluator(MetricRegistry(plugins)).evaluate(trace)
    }


def _bounded_gripper_spec():
    return ActionSpec(
        2,
        ActionRepresentation.OTHER,
        gripper_indices=(1,),
        rotation_representation=RotationRepresentation.NONE,
        gripper_convention=GripperConvention.CLOSED_POSITIVE,
        units=("normalized_command", "normalized_command"),
        minimum=np.array([-1, -1], dtype=np.float32),
        maximum=np.array([1, 1], dtype=np.float32),
        dtype="float32",
        control_frequency_hz=20.0,
    )


@pytest.mark.parametrize("length", (0, 1, 2))
def test_short_trace_availability_and_episode_length_are_not_zero_filled(length):
    actions = tuple((float(index), 0.0) for index in range(length))
    trace = synthetic_trace(
        actions,
        action_spec=metric_action_spec(),
        inference_durations=tuple(1_000_000 for _ in actions),
    )
    results = _evaluate(
        trace,
        EpisodeLengthMetric(), ActionVarianceMetric(), Smoothness1Metric(),
        Smoothness2Metric(), InferenceLatencyMetric(), ControlFrequencyMetric(),
    )
    assert results["episode.length"].status is MetricStatus.AVAILABLE
    assert results["episode.length"].value == length
    if length == 0:
        for metric_id in (
            "action.variance", "action.smoothness_1", "action.smoothness_2",
            "system.inference_latency", "system.control_frequency",
        ):
            assert results[metric_id].status is MetricStatus.UNAVAILABLE
            assert results[metric_id].value is None
    elif length == 1:
        assert results["action.variance"].status is MetricStatus.INSUFFICIENT_DATA
        assert results["action.smoothness_1"].status is MetricStatus.INSUFFICIENT_DATA
        assert results["action.smoothness_2"].status is MetricStatus.INSUFFICIENT_DATA
    else:
        assert results["action.variance"].status is MetricStatus.AVAILABLE
        assert results["action.smoothness_1"].status is MetricStatus.AVAILABLE
        assert results["action.smoothness_2"].status is MetricStatus.INSUFFICIENT_DATA


def test_long_trace_manual_oracle_covers_axes_reductions_gripper_validity_and_latency():
    # Per-dimension population variances are 5.25 and 1.0; aggregate is
    # their unweighted mean, 3.125. Smooth1 is the mean Euclidean norm of
    # three first differences. Smooth2 is the mean norm of two second differences.
    actions = ((0, -1), (1, 1), (3, -1), (6, 1))
    trace = synthetic_trace(
        actions,
        action_spec=_bounded_gripper_spec(),
        prediction_validities=(
            PredictionValidity.VALID,
            PredictionValidity.INVALID_VALUE,
            PredictionValidity.DECODE_ERROR,
            PredictionValidity.VALID,
        ),
        inference_durations=(1_000_000, 2_000_000, 4_000_000, 8_000_000),
        terminal_status=EpisodeTerminalStatus.SUCCESS,
        success_signal=True,
    )
    results = _evaluate(
        trace,
        TaskSuccessMetric(), ActionVarianceMetric(), Smoothness1Metric(), Smoothness2Metric(),
        GripperFlickerRateMetric(), InvalidPredictionRateMetric(), EpisodeLengthMetric(),
        InferenceLatencyMetric(), ControlFrequencyMetric(),
    )
    assert results["task.success"].value == 1
    assert results["task.success"].diagnostics["terminal_status"] == "success"
    assert results["action.variance"].diagnostics["per_dimension_variance"] == pytest.approx((5.25, 1.0))
    assert results["action.variance"].value == pytest.approx(3.125, abs=1e-12)
    assert results["action.smoothness_1"].value == pytest.approx(
        (math.sqrt(5) + math.sqrt(8) + math.sqrt(13)) / 3, abs=2e-7,
    )
    assert results["action.smoothness_2"].value == pytest.approx(
        (math.sqrt(17) + math.sqrt(17)) / 2, abs=2e-7,
    )
    flicker = results["failure.gripper_flicker_rate"]
    assert flicker.value == pytest.approx(2 / 3, abs=1e-12)
    assert flicker.diagnostics == {"transitions": 3, "flickers": 2, "unknown_steps": 0}
    invalid = results["failure.invalid_prediction_rate"]
    assert invalid.value == 0.5
    assert invalid.diagnostics["validity_counts"] == {
        "decode_error": 1, "invalid_value": 1, "valid": 2,
    }
    assert invalid.diagnostics["bounds_violation_count"] == 2
    assert invalid.diagnostics["nan_value_count"] == 0
    assert invalid.diagnostics["infinity_value_count"] == 0
    assert results["episode.length"].value == 4
    latency = results["system.inference_latency"]
    assert latency.value == 3.75
    assert latency.diagnostics == pytest.approx({
        "mean": 3.75,
        "median": 3.0,
        "p95": 7.4,
        "minimum": 1.0,
        "maximum": 8.0,
        "standard_deviation": math.sqrt(7.1875),
    })
    control = results["system.control_frequency"]
    assert control.value == 20.0 and control.unit == "Hz"
    assert control.diagnostics["measured"] is False


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_decoded_actions_are_rejected_before_a_trace_can_claim_zero(bad):
    spec = _bounded_gripper_spec()
    trace = synthetic_trace(((0, -1),), action_spec=spec)
    context = trace.step_contexts[0]
    with pytest.raises(ValueError, match="finite"):
        ActionPrediction(
            PredictionId("bad-value"), context.step_id,
            np.array([[bad, 0]], dtype=np.float32), spec, 1, 1, 1,
        )


def test_oracle_recomputation_is_semantically_identical_and_does_not_mutate_trace(tmp_path):
    from ovlab_runner import TraceCodec

    trace = synthetic_trace(((0, 0), (1, 1), (2, 0)), action_spec=_bounded_gripper_spec())
    plugins = (
        ActionVarianceMetric(), Smoothness1Metric(), Smoothness2Metric(),
        InvalidPredictionRateMetric(), EpisodeLengthMetric(), InferenceLatencyMetric(),
        ControlFrequencyMetric(),
    )
    before = tuple(action.applied_action.tobytes() for action in trace.executed_actions)
    online = MetricEvaluator(MetricRegistry(plugins)).evaluate(trace)
    path = tmp_path / "trace"
    TraceCodec().encode(trace, path)
    loaded = TraceCodec().decode(path)
    offline = MetricEvaluator(MetricRegistry(plugins)).evaluate(loaded)
    assert offline == online
    assert tuple(action.applied_action.tobytes() for action in trace.executed_actions) == before
