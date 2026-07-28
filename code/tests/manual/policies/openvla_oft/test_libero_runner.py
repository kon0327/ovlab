"""Gate E3-E5: official native OpenVLA-OFT through isolated RPC and LIBERO-10."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from ovlab_benchctl import ConfigResolver
from ovlab_benchmarks.libero import LiberoBenchmarkAdapter, LiberoRendererBackend
from ovlab_core.contracts import (
    EpisodeContext, EpisodeId, GripperConvention, Instruction, InstructionId, InstructionSource,
    RunContext, RunId, TaskId,
)
from ovlab_metrics import EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, MetricStatus
from ovlab_openvla_common import LiberoActionChunkCodec
from ovlab_openvla_oft import OpenVlaOftAdapter, OpenVlaOftRuntime
from ovlab_remote_policy import OwnedPolicyServiceProcess, RemotePolicyAdapter, UnixPolicyClient
from ovlab_runner import (
    ArtifactError, ExperimentRunner, FilesystemRunArtifactStore, ProvenanceSnapshot,
    StaticProvenanceProvider,
)

pytestmark = [pytest.mark.libero, pytest.mark.openvla, pytest.mark.oft, pytest.mark.gpu, pytest.mark.manual]

REPOSITORY = Path(__file__).resolve().parents[5]
RESOURCE_REVISION = "95220f9a3421a7ff12d4218e73d09ade830fa9a3"
OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
TASK_ID = TaskId("libero/10/0")


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _plain(value):
    """Remove contract immutability wrappers without changing recorded content."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _profile() -> Path:
    path = Path(os.environ["OVLAB_LOCAL_PROFILE"]).resolve()
    assert path.is_file()
    return path


def _resolve():
    return ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero10-openvla-oft-rpc-smoke.yaml",
        local_profile=_profile(), execution_profile="profiles/libero-bench-egl.yaml",
    )


def _record_fixed_libero10_observation(resolved):
    benchmark = LiberoBenchmarkAdapter(resolved.benchmark_settings)
    run = RunContext(RunId("gate-e-frozen-input"), 1, "Gate E frozen input", 42)
    try:
        benchmark.initialize(run)
        task = benchmark.list_tasks()[0]
        instruction = Instruction(
            InstructionId("gate-e-frozen-instruction"), task.natural_language_instruction, 2,
            InstructionSource.BENCHMARK,
        )
        episode = EpisodeContext(run.run_id, task.task_id, EpisodeId("gate-e-frozen-episode"), 0, 42, instruction)
        reset = benchmark.reset_episode(episode)
        observation = reset.initial_observation
        assert reset.metadata["initial_state_index"] == 0
        return run, episode, observation
    finally:
        benchmark.close()


def test_frozen_official_inference_matches_thin_adapter_stage_by_stage():
    assert os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    resolved = _resolve()
    run, episode, observation = _record_fixed_libero10_observation(resolved)
    source_checksums = {
        item.name: hashlib.sha256(item.data.tobytes()).hexdigest() for item in observation.images
    }
    source_checksums.update({
        item.name: hashlib.sha256(item.values.tobytes()).hexdigest() for item in observation.proprioception
    })
    runtime = OpenVlaOftRuntime()
    adapter = OpenVlaOftAdapter(replace(resolved.policy_settings, record_raw_output=True), runtime)
    try:
        capabilities = adapter.initialize(run)
        images = {item.name: item.data for item in observation.images}
        proprio = observation.proprioception[0].values
        official = runtime.predict(
            images["camera.primary.rgb"], images["camera.wrist.rgb"], proprio,
            observation.instruction.text,
        )
        official_final = LiberoActionChunkCodec().encode(official.decoded_actions)
        adapter.reset_episode(episode)
        wrapped = adapter.predict(observation)
        assert capabilities.metadata["method_descriptor"]["family"] == "openvla_oft"
        assert official.metadata["prompt"] == wrapped.metadata["runtime"]["prompt"]
        assert _plain(official.metadata["processor_calls"]) == _plain(
            wrapped.metadata["runtime"]["processor_calls"]
        )
        np.testing.assert_allclose(
            wrapped.raw_output.value, official.decoded_actions.value, rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(wrapped.actions, official_final, rtol=1e-5, atol=1e-6)
        assert wrapped.metadata["primary_rgb_sha256"] == _sha(images["camera.primary.rgb"])
        assert wrapped.metadata["wrist_rgb_sha256"] == _sha(images["camera.wrist.rgb"])
        assert wrapped.metadata["proprioception_sha256"] == _sha(proprio)
        assert wrapped.metadata["normalized_proprioception_sha256"] == _sha(
            official.normalized_proprioception
        )
        assert wrapped.actions.shape == (8, 7) and wrapped.actions.dtype == np.float32
        assert np.all(np.isfinite(wrapped.actions)) and np.all(np.abs(wrapped.actions) <= 1)
        assert wrapped.action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
        assert wrapped.metadata["codec_application_count_per_action"] == 1
        assert runtime.load_counts == {
            "backbone": 1, "processor": 1, "published_peft_adapter": 0,
            "action_head": 1, "proprio_projector": 1,
        }
        current_checksums = {
            item.name: hashlib.sha256(item.data.tobytes()).hexdigest() for item in observation.images
        }
        current_checksums.update({
            item.name: hashlib.sha256(item.values.tobytes()).hexdigest()
            for item in observation.proprioception
        })
        assert current_checksums == source_checksums
    finally:
        adapter.close()
    assert runtime._model is runtime._processor is runtime._action_head is runtime._proprio_projector is None


def _service(socket_path, log_path):
    return OwnedPolicyServiceProcess(
        [
            shutil.which("conda"), "run", "--no-capture-output", "-n", "openvla-oft",
            "python", "-m", "ovlab_openvla_oft.service", "--socket", str(socket_path),
            "--registry", str(REPOSITORY / "configs/resources/registry.yaml"),
            "--resource-id", "openvla-oft-7b-finetuned-libero-10", "--device", "cuda:0",
        ],
        socket_path, log_path,
        environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        startup_timeout_s=180, shutdown_timeout_s=20,
    )


def test_one_native_open_loop_chunk_rollout_over_isolated_rpc():
    assert os.environ.get("OVLAB_RUN_LIBERO_INTEGRATION") == "1"
    assert os.environ.get("CONDA_DEFAULT_ENV") == "openvla-oft"
    assert "prismatic" not in sys.modules
    resolved = _resolve()
    assert resolved.benchmark_settings.renderer.requested_backend is LiberoRendererBackend.EGL
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == "0"
    unique = uuid.uuid4().hex
    run = RunContext(RunId(f"gate-e-oft-rpc-{unique}"), 1, "Gate E native OFT RPC smoke", 42)
    root = Path(tempfile.mkdtemp(prefix=f"ovlab-{run.run_id}-", dir="/tmp"))
    os.chmod(root, 0o700)
    socket_path, log_path = root / "policy.sock", root / "service.log"
    owner = _service(socket_path, log_path)
    policy = RemotePolicyAdapter(
        UnixPolicyClient(socket_path, request_timeout_s=120),
        start_service=owner.start, stop_service=owner.stop,
        transport_metadata={
            "type": "AF_UNIX", "service_environment": "openvla-oft",
            "runner_environment": "openvla-oft", "service_log": str(log_path),
        },
    )
    store = FilesystemRunArtifactStore(resolved.artifact_settings.root)
    registry = MetricRegistry.default()
    runner = ExperimentRunner(
        resolved.create_plan(run, (TASK_ID,)), LiberoBenchmarkAdapter(resolved.benchmark_settings),
        policy, store, metric_registry=registry,
        provenance_provider=StaticProvenanceProvider(ProvenanceSnapshot(
            external_commits={"libero": LIBERO_COMMIT, "openvla-oft": OFT_COMMIT},
            environment_snapshot_reference="openvla-oft runner + isolated openvla-oft service",
            checkpoint_identity=f"moojink/openvla-7b-oft-finetuned-libero-10@{RESOURCE_REVISION}",
            dataset_identity="LIBERO-10/task-0/init-state-0",
        )),
        configuration_snapshot=resolved.configuration_snapshot(),
    )
    try:
        report = runner.connect()
        assert report.compatibility_report.compatible and report.compatibility_report.issues == ()
        assert report.policy_name == "ovlab-openvla-oft"
        requirements = policy.capabilities.observation_requirements
        assert [item.name for item in requirements.images] == ["camera.primary.rgb", "camera.wrist.rgb"]
        assert [item.name for item in requirements.proprioception] == ["robot.proprioception"]
        assert policy.capabilities.minimum_action_horizon == policy.capabilities.maximum_action_horizon == 8
        method = policy.handshake["method_descriptor"]
        assert method["family"] == "openvla_oft" and method["acronym_expansion"] == "optimized_fine_tuning"
        assert method["runtime_active_adapter"] is False and method["backbone_merge_status"] == "merged"
        assert method["film"] is method["diffusion"] is False and method["quantization"] == "none"
        assert method["qp_classification"] == "absent"
        assert method["load_counts"] == {
            "backbone": 1, "processor": 1, "published_peft_adapter": 0,
            "action_head": 1, "proprio_projector": 1,
        }
        assert policy.handshake["model_identity"]["revision"] == RESOURCE_REVISION
        runner.run()
    finally:
        runner.close()
        owner.stop()
    assert owner.process is None and not socket_path.exists()
    assert log_path.is_file() and "Traceback" not in log_path.read_text(errors="replace")

    run_path = store._run_path(run.run_id)
    episode_path = next(run_path.glob("tasks/*/episodes/*"))
    payload = json.loads((episode_path / "trace.json").read_text())
    trace = store.read_episode_trace(run.run_id, TASK_ID, payload["episode_context"]["episode_id"])
    stored = store.read_metric_results(run.run_id, TASK_ID, trace.episode_context.episode_id)
    plugins = tuple(
        registry.resolve(metric_id) for metric_id in runner.plan.enabled_metric_ids
        if isinstance(registry.resolve(metric_id), EpisodeMetricPlugin)
    )
    configurations = {
        key: value for key, value in runner.plan.metric_configurations.items()
        if key in {plugin.descriptor.metric_id for plugin in plugins}
    }
    assert MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, configurations) == stored
    assert len(trace.policy_predictions) == 1 and len(trace.executed_actions) == 3
    prediction = trace.policy_predictions[0]
    assert prediction.actions.shape == (8, 7) and prediction.actions.dtype == np.float32
    assert [action.selected_chunk_index for action in trace.executed_actions] == [0, 1, 2]
    audit = trace.metadata["action_chunk_audit"][0]
    assert audit["executed_offsets"] == (0, 1, 2)
    assert audit["discarded_unexecuted_offsets"] == (3, 4, 5, 6, 7)
    assert audit["amortized_generation_per_executed_action_ns"] > 0
    assert prediction.metadata["service_inference_duration_ns"] > 0
    assert prediction.metadata["rpc_round_trip_duration_ns"] >= prediction.metadata["service_inference_duration_ns"]
    assert all(action.metadata["action_application_duration_ns"] > 0 for action in trace.executed_actions)
    assert all(action.requested_action.tobytes() == action.applied_action.tobytes() for action in trace.executed_actions)
    assert all(len(observation.images) == 2 and len(observation.proprioception) == 1 for observation in trace.observations)
    collision = next(result for result in stored if result.metric_id == "failure.collision_rate")
    assert collision.status is MetricStatus.UNAVAILABLE and collision.value is None
    with pytest.raises(ArtifactError, match="already exists"):
        store.write_episode_trace(run.run_id, trace)
    started = json.loads((run_path / "manifest.started.json").read_text())
    assert started["scientific_config_hash"] == resolved.scientific_config_hash
    assert started["execution_config_hash"] == resolved.execution_config_hash
