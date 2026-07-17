"""Gate C: bounded real LIBERO + OpenVLA Vanilla runner integration."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ovlab_benchctl import ConfigResolver
from ovlab_benchmarks.libero import LiberoBenchmarkAdapter
from ovlab_core.contracts import PolicyObservation, RunContext, RunId, SignalAccess, TaskId
from ovlab_metrics import EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, MetricStatus
from ovlab_openvla_vanilla import HuggingFaceOpenVlaRuntime, OpenVlaVanillaAdapter
from ovlab_runner import (
    ExperimentRunner, FilesystemRunArtifactStore, ProvenanceSnapshot, StaticProvenanceProvider,
)

pytestmark = [pytest.mark.openvla, pytest.mark.libero, pytest.mark.gpu, pytest.mark.manual]

REPOSITORY = Path(__file__).resolve().parents[5]
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OPENVLA_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"


class CountingRuntime(HuggingFaceOpenVlaRuntime):
    def __init__(self):
        super().__init__()
        self.load_count = 0

    def load(self, settings):
        self.load_count += 1
        return super().load(settings)


class TrackingOpenVlaAdapter(OpenVlaVanillaAdapter):
    def __init__(self, settings, runtime):
        super().__init__(settings, runtime)
        self.seen_observations = []

    def _predict(self, observation):
        assert isinstance(observation, PolicyObservation)
        self.seen_observations.append(observation)
        return super()._predict(observation)


def _profile() -> Path:
    value = os.environ.get("OVLAB_LOCAL_PROFILE")
    if not value:
        pytest.skip("set OVLAB_LOCAL_PROFILE to an explicit local resource profile")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"OVLAB_LOCAL_PROFILE does not exist: {path}")
    return path


def test_one_bounded_libero_rollout_records_recomputable_trace():
    if os.environ.get("OVLAB_RUN_LIBERO_INTEGRATION") != "1":
        pytest.skip("set OVLAB_RUN_LIBERO_INTEGRATION=1 for the real bounded rollout")
    resolved = ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero-vanilla-smoke.yaml", local_profile=_profile()
    )
    run = RunContext(
        RunId(os.environ.get("OVLAB_RUN_ID", "gate-c-libero-openvla-vanilla")), 1,
        "bounded LIBERO OpenVLA Vanilla integration reference", 42,
    )
    plan = resolved.create_plan(run, (TaskId("libero/spatial/0"),))
    benchmark = LiberoBenchmarkAdapter(resolved.benchmark_settings)
    runtime = CountingRuntime()
    assert runtime._model is None and runtime._processor is None and runtime.load_count == 0
    policy = TrackingOpenVlaAdapter(resolved.policy_settings, runtime)
    store = FilesystemRunArtifactStore(resolved.artifact_settings.root)
    registry = MetricRegistry.default()
    checkpoint = resolved.policy_settings.model
    runner = ExperimentRunner(
        plan, benchmark, policy, store, metric_registry=registry,
        provenance_provider=StaticProvenanceProvider(ProvenanceSnapshot(
            external_commits={"libero": LIBERO_COMMIT, "openvla": OPENVLA_COMMIT},
            environment_snapshot_reference=os.environ.get("CONDA_DEFAULT_ENV"),
            checkpoint_identity=f"{checkpoint.source}@{checkpoint.revision}",
            dataset_identity="LIBERO-Spatial/task-0/init-state-0",
        )),
        configuration_snapshot=resolved.configuration_snapshot(),
    )
    report = runner.connect()
    assert report.compatibility_report.compatible
    assert runtime.load_count == 1
    policy_metadata = report.metadata["policy_capabilities"]
    assert policy_metadata["policy_family"] == "openvla-vanilla"
    assert "qp" not in json.dumps(policy_metadata).lower()
    assert policy_metadata["checkpoint_identity"]["openvla_git_commit"] == OPENVLA_COMMIT
    assert policy_metadata["checkpoint_identity"]["snapshot_revision"] == checkpoint.revision
    assert policy.capabilities.minimum_action_horizon == policy.capabilities.maximum_action_horizon == 1
    assert not policy.capabilities.supports_action_chunks
    runner.run()

    assert runtime.load_count == 1
    assert runtime._model is None and runtime._processor is None
    assert policy.seen_observations
    assert all(
        "benchmark.task_success" not in {item.name for item in observation.images + observation.proprioception}
        for observation in policy.seen_observations
    )
    run_path = store._run_path(run.run_id)
    episode_path = next(run_path.glob("tasks/*/episodes/*"))
    payload = json.loads((episode_path / "trace.json").read_text(encoding="utf-8"))
    episode_id = payload["episode_context"]["episode_id"]
    trace = store.read_episode_trace(run.run_id, TaskId("libero/spatial/0"), episode_id)
    stored = store.read_metric_results(run.run_id, TaskId("libero/spatial/0"), episode_id)
    plugins = tuple(
        registry.resolve(metric_id) for metric_id in plan.enabled_metric_ids
        if isinstance(registry.resolve(metric_id), EpisodeMetricPlugin)
    )
    configurations = {
        key: value for key, value in plan.metric_configurations.items()
        if key in {plugin.descriptor.metric_id for plugin in plugins}
    }
    assert MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, configurations) == stored
    assert len(trace.policy_predictions) == len(trace.executed_actions) == 2
    assert all(prediction.horizon == 1 and prediction.inference_duration_ns > 0
               for prediction in trace.policy_predictions)
    assert all(prediction.actions.shape == (1, 7) and np.all(np.isfinite(prediction.actions))
               for prediction in trace.policy_predictions)
    assert all(action.selected_chunk_index == 0 for action in trace.executed_actions)
    assert all(action.requested_action.tobytes() == action.applied_action.tobytes()
               for action in trace.executed_actions)
    assert all(action.metadata["closed_loop_step_duration_ns"] > 0 for action in trace.executed_actions)
    assert all("rpc" not in json.dumps(dict(action.metadata)).lower() for action in trace.executed_actions)
    success_signals = tuple(
        signal for signal in trace.evaluation_signals if signal.name == "benchmark.task_success"
    )
    assert success_signals and all(signal.access is SignalAccess.EVALUATION_ONLY for signal in success_signals)
    success = next(result for result in stored if result.metric_id == "task.success")
    assert success.status is MetricStatus.AVAILABLE and success.value == int(success_signals[-1].value)
    collision = next(result for result in stored if result.metric_id == "failure.collision_rate")
    assert collision.status is MetricStatus.UNAVAILABLE and collision.value is None
    connection = json.loads((run_path / "connection.json").read_text(encoding="utf-8"))
    started = json.loads((run_path / "manifest.started.json").read_text(encoding="utf-8"))
    completed = json.loads((run_path / "manifest.completed.json").read_text(encoding="utf-8"))
    assert connection["metadata"]["benchmark_capabilities"]["libero_commit"] == LIBERO_COMMIT
    assert connection["metadata"]["policy_capabilities"]["prompt_template"] == "openvla-v1@1.0.0"
    assert connection["metadata"]["policy_capabilities"]["action_codec"] == "openvla-to-libero@1.0.0"
    assert started["scientific_config_hash"] == resolved.scientific_config_hash
    assert started["execution_config_hash"] == resolved.execution_config_hash
    assert completed["status"] == "completed" and completed["episode_count"] == 1
