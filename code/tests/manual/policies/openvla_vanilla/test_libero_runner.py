"""Bounded real LIBERO + OpenVLA + runner integration (explicit opt-in)."""

import os

import pytest

from ovlab_benchmarks.libero import LiberoAdapterSettings, LiberoBenchmarkAdapter
from ovlab_core.contracts import RunContext, RunId, SignalAccess, TaskId
from ovlab_metrics import MetricEvaluator, MetricRegistry
from ovlab_openvla_common import OpenVlaModelSource
from ovlab_openvla_vanilla import OpenVlaVanillaAdapter, OpenVlaVanillaSettings
from ovlab_runner import (
    ActionExecutionMode, ActionExecutionPolicy, ExperimentPlan, ExperimentRunner,
    InMemoryRunArtifactStore, TraceRecordingPolicy,
)

pytestmark = [pytest.mark.openvla, pytest.mark.libero, pytest.mark.gpu, pytest.mark.manual]


def test_one_bounded_libero_rollout_records_recomputable_trace():
    if os.environ.get("OVLAB_RUN_LIBERO_INTEGRATION") != "1":
        pytest.skip("set OVLAB_RUN_LIBERO_INTEGRATION=1 for the real bounded rollout")
    checkpoint = os.environ.get("OVLAB_OPENVLA_CHECKPOINT")
    key = os.environ.get("OVLAB_OPENVLA_UNNORM_KEY")
    if not checkpoint or not key:
        pytest.skip("set OVLAB_OPENVLA_CHECKPOINT and OVLAB_OPENVLA_UNNORM_KEY")

    run = RunContext(RunId("manual-libero-openvla"), 1, "bounded LIBERO OpenVLA integration", 7)
    benchmark = LiberoBenchmarkAdapter(LiberoAdapterSettings(
        suite_names=("LIBERO-10",), task_indices=(0,), maximum_episode_steps=2,
        initialization_settling_steps=1,
    ))
    policy = OpenVlaVanillaAdapter(OpenVlaVanillaSettings(OpenVlaModelSource(checkpoint), key))
    plan = ExperimentPlan(
        run, (TaskId("libero/10/0"),), 1, 7, 2,
        action_execution_policy=ActionExecutionPolicy(ActionExecutionMode.RECEDING_HORIZON),
        enabled_metric_ids=("task.success", "system.inference_latency"),
        trace_recording_policy=TraceRecordingPolicy(record_privileged_signals=False),
    )
    store = InMemoryRunArtifactStore()
    runner = ExperimentRunner(plan, benchmark, policy, store)
    report = runner.connect()
    assert report.compatibility_report.compatible
    runner.run()

    episode = next(iter(store.runs[str(run.run_id)]["episodes"].values()))
    trace = episode["trace"]
    assert trace.policy_predictions and len(trace.policy_predictions) == len(trace.executed_actions)
    assert all(prediction.horizon == 1 and prediction.inference_duration_ns > 0
               for prediction in trace.policy_predictions)
    assert all(image.name == "camera.primary.rgb" for observation in trace.observations for image in observation.images)
    assert all(signal.access is not SignalAccess.PRIVILEGED for signal in trace.evaluation_signals)
    assert "completed" in store.runs[str(run.run_id)]

    plugins = [MetricRegistry.default().resolve(name) for name in ("task.success", "system.inference_latency")]
    recomputed = MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, {})
    stored = episode["metrics"]
    assert tuple(result.metric_id for result in recomputed) == tuple(result.metric_id for result in stored)
