"""Gate B: opt-in pinned LIBERO plus deterministic dependency-free policy."""

import os
from pathlib import Path
import json
import hashlib

import numpy as np
import pytest

from ovlab_benchctl import ConfigResolver, load
from ovlab_benchmarks.libero import LiberoBenchmarkAdapter
from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION, ActionPrediction, ActionRepresentation, ColorSpace, EpisodeTerminalStatus,
    GripperConvention,
    ImageEncoding, ImageObservationSpec, ObservationRequirements, PolicyCapabilities,
    AdapterState, PolicyObservation, PredictionId, RotationRepresentation, RunContext, RunId,
    SignalAccess, TaskId,
)
from ovlab_metrics import EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, MetricStatus
from ovlab_policy_sdk import PolicyAdapter
from ovlab_runner import (
    ArtifactError, ExperimentRunner, FilesystemRunArtifactStore, ProvenanceSnapshot,
    RunnerState, StaticProvenanceProvider,
)

pytestmark = [
    pytest.mark.libero,
    pytest.mark.gpu,
    pytest.mark.manual,
    pytest.mark.skipif(
        os.environ.get("OVLAB_RUN_LIBERO_RUNNER") != "1",
        reason="explicit real LIBERO runner opt-in required",
    ),
]

REPOSITORY = Path(__file__).resolve().parents[4]
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OPENVLA_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"


class NoOpLiberoPolicy(PolicyAdapter):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.observations = []
        self.closed = False

    def _initialize(self, run_context):
        del run_context
        image = ImageObservationSpec(
            self.settings.camera_name,
            ((256, 256, 3),),
            "uint8",
            (ImageEncoding.RAW,),
            (ColorSpace.RGB,),
        )
        return PolicyCapabilities(
            "deterministic-libero-noop", "1.0.0", OVLAB_CONTRACT_VERSION,
            ObservationRequirements((image,), (), 1, 1, 0, 0), self.settings.action_spec,
            True, False, 1, 1, False, True, False,
            metadata={"policy_family": "mock", "integration_gate": "B"},
        )

    def _reset_episode(self, episode_context):
        del episode_context
        self.index = 0

    def _predict(self, observation):
        assert isinstance(observation, PolicyObservation)
        self.observations.append(observation)
        action = np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
        result = ActionPrediction(
            PredictionId(f"noop-{self.index}"), observation.step_id, action,
            self.capabilities.output_action_spec, observation.timestamp_ns + 1, 1, 1,
        )
        self.index += 1
        return result

    def _close(self):
        self.closed = True


def _profile() -> Path:
    value = os.environ.get("OVLAB_LOCAL_PROFILE")
    if not value:
        pytest.skip("set OVLAB_LOCAL_PROFILE to an explicit local resource profile")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"OVLAB_LOCAL_PROFILE does not exist: {path}")
    return path


def _array_references(value):
    if isinstance(value, dict):
        if "$array" in value:
            yield value
        for item in value.values():
            yield from _array_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _array_references(item)


def test_real_libero_runner_produces_recomputable_immutable_trace() -> None:
    resolved = ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero-mock-smoke.yaml",
        local_profile=_profile(),
        execution_profile="profiles/libero-bench-egl.yaml",
    )
    run = RunContext(RunId(os.environ.get("OVLAB_RUN_ID", "gate-b-libero-mock")), 1, "Gate B LIBERO mock smoke", 42)
    plan = resolved.create_plan(run, (TaskId("libero/spatial/0"),))
    benchmark = LiberoBenchmarkAdapter(resolved.benchmark_settings)
    policy = NoOpLiberoPolicy(resolved.policy_settings)
    store = FilesystemRunArtifactStore(resolved.artifact_settings.root)
    registry = MetricRegistry.default()
    runner = ExperimentRunner(
        plan, benchmark, policy, store, metric_registry=registry,
        provenance_provider=StaticProvenanceProvider(ProvenanceSnapshot(
            external_commits={"libero": LIBERO_COMMIT, "openvla": OPENVLA_COMMIT},
            environment_snapshot_reference="conda:openvla-oft",
        )),
        configuration_snapshot=resolved.configuration_snapshot(),
    )
    report = runner.connect()
    assert report.compatibility_report.compatible
    assert tuple(str(task.task_id) for task in report.selected_tasks) == ("libero/spatial/0",)
    assert report.metadata["benchmark_capabilities"]["libero_commit"] == LIBERO_COMMIT
    benchmark_caps = benchmark.capabilities
    assert benchmark_caps.task_suites == ("LIBERO-Spatial",)
    assert not benchmark_caps.supports_dynamic_instructions
    assert benchmark_caps.observation_spec.images[0].name == "camera.primary.rgb"
    assert benchmark_caps.observation_spec.images[0].shapes == ((256, 256, 3),)
    assert benchmark_caps.observation_spec.images[0].dtype == "uint8"
    assert benchmark_caps.observation_spec.metadata["image_transform"] == "rotate_180"
    action_spec = benchmark_caps.action_spec
    assert action_spec.dimension == 7
    assert action_spec.representation is ActionRepresentation.DELTA_POSE
    assert action_spec.translation_indices == (0, 1, 2)
    assert action_spec.rotation_indices == (3, 4, 5)
    assert action_spec.rotation_representation is RotationRepresentation.AXIS_ANGLE
    assert action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
    assert action_spec.units == ("normalized_command",) * 7
    np.testing.assert_array_equal(action_spec.minimum, np.full(7, -1, dtype=np.float32))
    np.testing.assert_array_equal(action_spec.maximum, np.full(7, 1, dtype=np.float32))
    runner.run()
    assert runner.state is RunnerState.COMPLETED
    assert policy.closed
    assert benchmark.state is AdapterState.CLOSED

    task = report.selected_tasks[0]
    run_path = store._run_path(run.run_id)
    started = json.loads((run_path / "manifest.started.json").read_text(encoding="utf-8"))
    completed = json.loads((run_path / "manifest.completed.json").read_text(encoding="utf-8"))
    connection = json.loads((run_path / "connection.json").read_text(encoding="utf-8"))
    source = load(run_path / "source_config.yaml")
    preserved = load(run_path / "resolved_config.yaml")
    assert source["kind"] == "scientific_experiment"
    assert preserved["scientific_config_hash"] == resolved.scientific_config_hash
    assert preserved["execution_config_hash"] == resolved.execution_config_hash
    assert started["scientific_config_hash"] == resolved.scientific_config_hash
    assert started["execution_config_hash"] == resolved.execution_config_hash
    assert started["recording_policy_hash"] == plan.trace_recording_policy.hash
    assert started["provenance"]["external_commits"]["libero"] == LIBERO_COMMIT
    assert started["benchmark"] == {"name": "libero-benchmark-adapter", "version": "0.2.0"}
    assert started["policy"] == {"name": "deterministic-libero-noop", "version": "1.0.0"}
    started_renderer = started["runtime"]["benchmark"]["libero_renderer"]
    assert started_renderer["requested_backend"] == "egl"
    assert started_renderer["resolved_backend"] == "egl"
    assert started_renderer["device_id"] == 0
    assert started_renderer["detected_renderer"] is None
    assert connection["compatible"] is True and connection["compatibility_issues"] == []
    assert connection["metadata"]["benchmark_capabilities"]["libero_commit"] == LIBERO_COMMIT
    assert completed["status"] == "completed"
    assert completed["episode_count"] == 1
    assert completed["episode_counts_by_terminal_status"] == {"time_limit": 1}
    completed_renderer = completed["metadata"]["benchmark_runtime"]["libero_renderer"]
    assert completed_renderer["requested_backend"] == "egl"
    assert completed_renderer["resolved_backend"] == "egl"
    assert completed_renderer["device_id"] == 0
    assert completed_renderer["detected_renderer"]["renderer"]
    episode_dirs = tuple(store._task_path(run.run_id, task.task_id).glob("episodes/*"))
    assert len(episode_dirs) == 1
    episode_id = next(iter(store._task_path(run.run_id, task.task_id).glob("episodes/*/trace.json"))).parent
    payload = json.loads((episode_id / "trace.json").read_text(encoding="utf-8"))
    references = tuple(_array_references(payload))
    assert references
    for reference in references:
        target = episode_id / reference["$array"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == reference["sha256"]
    assert (episode_id / "trace.finalized.json").is_file()
    assert (episode_id / "finalized.json").is_file()
    original_episode_id = payload["episode_context"]["episode_id"]
    trace = store.read_episode_trace(run.run_id, task.task_id, original_episode_id)
    stored = store.read_metric_results(run.run_id, task.task_id, original_episode_id)
    plugins = tuple(
        registry.resolve(metric_id) for metric_id in plan.enabled_metric_ids
        if isinstance(registry.resolve(metric_id), EpisodeMetricPlugin)
    )
    configs = {key: value for key, value in plan.metric_configurations.items() if key in {
        plugin.descriptor.metric_id for plugin in plugins
    }}
    recomputed = MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, configs)
    assert recomputed == stored
    assert trace.terminal_status is EpisodeTerminalStatus.TIME_LIMIT
    assert trace.instruction_events == (trace.episode_context.initial_instruction,)
    assert trace.policy_predictions and len(trace.policy_predictions) == len(trace.executed_actions)
    predictions = {prediction.prediction_id: prediction for prediction in trace.policy_predictions}
    assert len(predictions) == len(trace.policy_predictions)
    assert {action.prediction_id for action in trace.executed_actions} == set(predictions)
    assert all(action.selected_chunk_index == 0 for action in trace.executed_actions)
    assert all(
        action.requested_action.tobytes()
        == predictions[action.prediction_id].actions[action.selected_chunk_index].tobytes()
        for action in trace.executed_actions
    )
    assert all(action.requested_action.tobytes() == action.applied_action.tobytes() for action in trace.executed_actions)
    assert all(np.all(np.isfinite(action.applied_action)) for action in trace.executed_actions)
    assert all(action.metadata["closed_loop_step_duration_ns"] > 0 for action in trace.executed_actions)
    assert all(prediction.inference_duration_ns == 1 for prediction in trace.policy_predictions)
    assert all(observation.images[0].data.shape == (256, 256, 3) for observation in trace.observations)
    assert all(observation.images[0].data.dtype == np.uint8 for observation in trace.observations)
    assert all(observation.images[0].color_space is ColorSpace.RGB for observation in trace.observations)
    assert all(observation.images[0].metadata["transform"] == "rotate_180" for observation in trace.observations)
    assert all(observation.metadata == {} for observation in policy.observations)
    assert all(observation.proprioception == () for observation in policy.observations)
    assert all(tuple(image.name for image in observation.images) == ("camera.primary.rgb",) for observation in policy.observations)
    assert all(observation.instruction == trace.episode_context.initial_instruction for observation in policy.observations)
    assert all(
        "benchmark.task_success" not in {item.name for item in observation.images + observation.proprioception}
        for observation in policy.observations
    )
    success_signals = tuple(signal for signal in trace.evaluation_signals if signal.name == "benchmark.task_success")
    assert success_signals and all(signal.access is SignalAccess.EVALUATION_ONLY for signal in success_signals)
    assert all(signal.source == "libero" for signal in success_signals)
    assert success_signals[-1].value is False
    assert any(signal.access is SignalAccess.PRIVILEGED for signal in trace.evaluation_signals)
    terminal_signals = {signal.name: signal.value for signal in trace.evaluation_signals[-9:]}
    assert terminal_signals["episode.terminated"] is False
    assert terminal_signals["episode.truncated"] is True
    success_metric = next(result for result in stored if result.metric_id == "task.success")
    assert success_metric.status is MetricStatus.AVAILABLE
    assert success_metric.value == int(success_signals[-1].value)
    collision = next(result for result in stored if result.metric_id == "failure.collision_rate")
    assert collision.status is MetricStatus.UNAVAILABLE and collision.value is None
    with pytest.raises(ArtifactError, match="already exists"):
        store.write_episode_trace(run.run_id, trace)
