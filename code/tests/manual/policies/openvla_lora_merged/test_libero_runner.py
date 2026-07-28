"""Gate D4-D5: LIBERO-10 runner and isolated merged OpenVLA-LoRA service."""

from __future__ import annotations

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
    GripperConvention,
    ImageEncoding,
    RunContext,
    RunId,
    RotationRepresentation,
    SignalAccess,
    TaskId,
)
from ovlab_metrics import EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, MetricStatus
from ovlab_openvla_common import action_specs_match
from ovlab_remote_policy import OwnedPolicyServiceProcess, RemotePolicyAdapter, UnixPolicyClient
from ovlab_runner import (
    ArtifactError, ExperimentRunner,
    FilesystemRunArtifactStore,
    ProvenanceSnapshot,
    StaticProvenanceProvider,
)

pytestmark = [
    pytest.mark.openvla, pytest.mark.libero, pytest.mark.lora, pytest.mark.gpu, pytest.mark.manual,
]

REPOSITORY = Path(__file__).resolve().parents[5]
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OPENVLA_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"
CHECKPOINT = "openvla/openvla-7b-finetuned-libero-10"
CHECKPOINT_REVISION = "80970322773f81baa2e22fe495d0487b93a05cfa"
UNNORM_KEY = "libero_10"
TASK_ID = TaskId("libero/10/0")


def _profile() -> Path:
    value = os.environ.get("OVLAB_LOCAL_PROFILE")
    if not value:
        pytest.fail("Gate D requires OVLAB_LOCAL_PROFILE with local LIBERO and artifact paths")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"OVLAB_LOCAL_PROFILE does not exist: {path}")
    return path


def _service_process(socket_path: Path, log_path: Path) -> OwnedPolicyServiceProcess:
    conda = shutil.which("conda")
    if conda is None:
        pytest.fail("conda executable is required to launch the isolated openvla policy service")
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "openvla",
        "python",
        "-m",
        "ovlab_openvla_lora_merged.service",
        "--socket",
        str(socket_path),
        "--registry",
        str(REPOSITORY / "configs/resources/registry.yaml"),
        "--resource-id",
        "openvla-7b-finetuned-libero-10",
        "--unnorm-key",
        UNNORM_KEY,
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--attention-implementation",
        "flash_attention_2",
    ]
    return OwnedPolicyServiceProcess(
        command,
        socket_path,
        log_path,
        environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        startup_timeout_s=60,
        shutdown_timeout_s=15,
    )


def test_one_bounded_libero10_rollout_over_isolated_rpc():
    if os.environ.get("OVLAB_RUN_LIBERO_INTEGRATION") != "1":
        pytest.fail("Gate D is an explicit real test; set OVLAB_RUN_LIBERO_INTEGRATION=1")
    assert os.environ.get("CONDA_DEFAULT_ENV") == "openvla-oft"
    assert "ovlab_openvla_oft" not in sys.modules
    resolved = ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero10-lora-merged-rpc-smoke.yaml",
        local_profile=_profile(),
        execution_profile="profiles/libero-bench-egl.yaml",
    )
    renderer = resolved.benchmark_settings.renderer
    assert renderer.requested_backend is LiberoRendererBackend.EGL
    assert renderer.resolved_backend is LiberoRendererBackend.EGL
    assert os.environ.get("MUJOCO_GL") == "egl"
    assert os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(renderer.device_id)

    unique = uuid.uuid4().hex
    run_id = f"gate-d-lora-merged-rpc-{unique}"
    service_root = Path(tempfile.mkdtemp(prefix=f"ovlab-{run_id}-", dir="/tmp"))
    os.chmod(service_root, 0o700)
    socket_path = service_root / "policy.sock"
    log_path = service_root / "openvla-service.log"
    owner = _service_process(socket_path, log_path)
    client = UnixPolicyClient(socket_path, request_timeout_s=90)
    policy = RemotePolicyAdapter(
        client,
        start_service=owner.start,
        stop_service=owner.stop,
        transport_metadata={
            "type": "AF_UNIX",
            "framing": "uint32-be-length-prefixed-json",
            "service_environment": "openvla",
            "runner_environment": "openvla-oft",
            "service_log": str(log_path),
        },
    )
    run = RunContext(
        RunId(run_id),
        1,
        "bounded LIBERO-10 merged OpenVLA-LoRA methodological RPC reference",
        42,
    )
    plan = resolved.create_plan(run, (TASK_ID,))
    benchmark = LiberoBenchmarkAdapter(resolved.benchmark_settings)
    store = FilesystemRunArtifactStore(resolved.artifact_settings.root)
    registry = MetricRegistry.default()
    runner = ExperimentRunner(
        plan,
        benchmark,
        policy,
        store,
        metric_registry=registry,
        provenance_provider=StaticProvenanceProvider(
            ProvenanceSnapshot(
                external_commits={"libero": LIBERO_COMMIT, "openvla": OPENVLA_COMMIT},
                environment_snapshot_reference="openvla-oft runner + openvla policy service",
                checkpoint_identity=f"{CHECKPOINT}@{CHECKPOINT_REVISION}",
                dataset_identity="LIBERO-10/task-0/init-state-0",
            )
        ),
        configuration_snapshot=resolved.configuration_snapshot(),
    )
    try:
        report = runner.connect()
        assert report.compatibility_report.compatible
        assert report.compatibility_report.issues == ()
        assert report.policy_name == "ovlab-openvla-lora-merged"
        assert policy.capabilities.minimum_action_horizon == policy.capabilities.maximum_action_horizon == 1
        requirements = policy.capabilities.observation_requirements
        assert requirements.proprioception == ()
        assert len(requirements.images) == 1
        assert requirements.images[0].name == "camera.primary.rgb"
        assert requirements.images[0].shapes == ((256, 256, 3),)
        assert requirements.images[0].dtype == "uint8"
        assert requirements.images[0].encodings == (ImageEncoding.RAW,)
        spec = policy.capabilities.output_action_spec
        assert spec.dimension == 7
        assert spec.translation_indices == (0, 1, 2)
        assert spec.rotation_indices == (3, 4, 5)
        assert spec.rotation_representation is RotationRepresentation.AXIS_ANGLE
        assert spec.gripper_indices == (6,)
        assert spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
        np.testing.assert_array_equal(spec.minimum, np.full(7, -1.0, dtype=np.float32))
        np.testing.assert_array_equal(spec.maximum, np.full(7, 1.0, dtype=np.float32))
        assert spec.units == ("normalized_command",) * 7
        assert action_specs_match(spec, benchmark.capabilities.action_spec)
        handshake = policy.handshake
        assert handshake["protocol_version"] == "ovlab-policy-rpc/1.0.0"
        assert handshake["model_identity"]["configured_source"] == CHECKPOINT
        assert handshake["model_identity"]["snapshot_revision"] == CHECKPOINT_REVISION
        assert handshake["model_identity"]["openvla_git_commit"] == OPENVLA_COMMIT
        assert handshake["model_identity"]["expected_checksum"] == (
            "33abee128d94d3bf54660b48c233c525f7969608924c58152602acca1b190eed"
        )
        assert handshake["normalization_identity"]["unnorm_key"] == UNNORM_KEY
        assert handshake["normalization_identity"]["action_statistics_identity"].startswith("sha256:")
        assert handshake["prompt_template_identity"] == "openvla-v1@1.0.0"
        codec = handshake["action_codec_identity"]
        assert codec["conversion_owner"] == "OpenVlaMergedLoraAdapter"
        assert codec["application_count"] == 1
        assert codec["output_gripper_convention"] == "closed_positive"
        assert handshake["runtime_versions"]["torch"] == "2.2.0"
        assert handshake["runtime_versions"]["transformers"] == "4.40.1"
        assert handshake["runtime_versions"]["flash_attn"] == "2.5.5"
        method = handshake["method_descriptor"]
        assert method["family"] == "lora"
        assert method["artifact_form"] == "merged_full_weights"
        assert method["merge_status"] == "merged"
        assert method["active_peft_adapter"] is False
        assert method["runtime_peft_modules"] is False
        assert method["quantization"] == "none"
        assert method["qp_profile"] is None
        assert method["load_counts"] == {"model": 1, "processor": 1, "peft_adapter": 0}
        assert method["total_runtime_parameter_count"] > 0
        assert method["runtime_parameter_trainability"] == "irrelevant"
        assert method["lora_configuration"]["rank"] == 32
        assert method["lora_configuration"]["alpha"] == 16
        assert method["lora_configuration"]["scaling"] == 0.5
        assert method["lora_configuration"]["target_policy"] == "all-linear"
        runner.run()
    finally:
        runner.close()
        owner.stop()

    assert owner.process is None
    assert not socket_path.exists()
    assert log_path.is_file()
    service_log = log_path.read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in service_log

    run_path = store._run_path(run.run_id)
    episode_path = next(run_path.glob("tasks/*/episodes/*"))
    payload = json.loads((episode_path / "trace.json").read_text(encoding="utf-8"))
    episode_id = payload["episode_context"]["episode_id"]
    trace = store.read_episode_trace(run.run_id, TASK_ID, episode_id)
    stored = store.read_metric_results(run.run_id, TASK_ID, episode_id)
    plugins = tuple(
        registry.resolve(metric_id)
        for metric_id in plan.enabled_metric_ids
        if isinstance(registry.resolve(metric_id), EpisodeMetricPlugin)
    )
    configurations = {
        key: value
        for key, value in plan.metric_configurations.items()
        if key in {plugin.descriptor.metric_id for plugin in plugins}
    }
    assert MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, configurations) == stored
    assert len(trace.policy_predictions) == len(trace.executed_actions) == 2
    for prediction in trace.policy_predictions:
        assert prediction.actions.dtype == np.float32
        assert prediction.actions.shape == (1, 7)
        assert prediction.action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
        assert prediction.inference_duration_ns == prediction.metadata["service_inference_duration_ns"] > 0
        assert prediction.metadata["rpc_round_trip_duration_ns"] >= prediction.inference_duration_ns
        assert prediction.metadata["rpc_protocol_version"] == "ovlab-policy-rpc/1.0.0"
    assert all(action.selected_chunk_index == 0 for action in trace.executed_actions)
    assert all(
        action.requested_action.tobytes() == action.applied_action.tobytes()
        for action in trace.executed_actions
    )
    assert all(action.metadata["closed_loop_step_duration_ns"] > 0 for action in trace.executed_actions)
    assert all(observation.metadata == {} for observation in trace.observations)
    assert all(observation.proprioception == () for observation in trace.observations)
    assert all(
        observation.images[0].metadata["transform"] == "rotate_180"
        for observation in trace.observations
    )
    success_signals = tuple(
        signal for signal in trace.evaluation_signals if signal.name == "benchmark.task_success"
    )
    assert success_signals
    assert all(signal.access is SignalAccess.EVALUATION_ONLY for signal in success_signals)
    success = next(result for result in stored if result.metric_id == "task.success")
    assert success.status is MetricStatus.AVAILABLE
    assert success.value == int(success_signals[-1].value)
    collision = next(result for result in stored if result.metric_id == "failure.collision_rate")
    assert collision.status is MetricStatus.UNAVAILABLE and collision.value is None

    connection = json.loads((run_path / "connection.json").read_text(encoding="utf-8"))
    started = json.loads((run_path / "manifest.started.json").read_text(encoding="utf-8"))
    completed = json.loads((run_path / "manifest.completed.json").read_text(encoding="utf-8"))
    assert connection["metadata"]["benchmark_capabilities"]["libero_commit"] == LIBERO_COMMIT
    remote_manifest = connection["metadata"]["policy_capabilities"]["remote_policy"]
    assert remote_manifest["model_identity"]["snapshot_revision"] == CHECKPOINT_REVISION
    assert remote_manifest["normalization_identity"]["unnorm_key"] == UNNORM_KEY
    assert remote_manifest["transport"]["service_log"] == str(log_path)
    assert remote_manifest["method_descriptor"] == handshake["method_descriptor"]
    assert started["scientific_config_hash"] == resolved.scientific_config_hash
    assert started["execution_config_hash"] == resolved.execution_config_hash
    assert started["runtime"]["benchmark"]["libero_renderer"]["resolved_backend"] == "egl"
    assert started["runtime"]["policy"]["remote_policy"]["protocol_version"] == "ovlab-policy-rpc/1.0.0"
    completed_renderer = completed["metadata"]["benchmark_runtime"]["libero_renderer"]
    assert completed_renderer["resolved_backend"] == "egl"
    assert completed_renderer["detected_renderer"]["renderer"]
    assert completed["metadata"]["policy_runtime"]["remote_policy"]["model_identity"]["snapshot_revision"] == CHECKPOINT_REVISION
    assert completed["status"] == "completed" and completed["episode_count"] == 1
    with pytest.raises(ArtifactError, match="already exists"):
        store.write_episode_trace(run.run_id, trace)
