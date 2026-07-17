"""Trace codec and filesystem artifact immutability tests."""

import json

import pytest

from helpers.metric_traces import synthetic_trace
from ovlab_runner import ArtifactError, FilesystemRunArtifactStore, RunConfigurationSnapshot, TraceCodec
from ovlab_runner.artifacts.layout import safe_key


def test_trace_codec_round_trip_preserves_arrays_and_signal_access(tmp_path) -> None:
    trace = synthetic_trace(collision_values=(False, True), horizon=2)
    path = tmp_path / "episode"
    TraceCodec().encode(trace, path)
    decoded = TraceCodec().decode(path)
    assert decoded.terminal_status == trace.terminal_status
    assert decoded.episode_context == trace.episode_context
    assert decoded.policy_predictions[0].actions.dtype == trace.policy_predictions[0].actions.dtype
    assert decoded.policy_predictions[0].actions.shape == trace.policy_predictions[0].actions.shape
    assert decoded.policy_predictions[0].actions.tobytes() == trace.policy_predictions[0].actions.tobytes()
    assert decoded.evaluation_signals[-1].access == trace.evaluation_signals[-1].access


def test_trace_codec_detects_modified_and_missing_arrays(tmp_path) -> None:
    trace = synthetic_trace()
    modified = tmp_path / "modified"
    TraceCodec().encode(trace, modified)
    array = next((modified / "arrays").glob("*.npy"))
    array.write_bytes(array.read_bytes() + b"tampered")
    with pytest.raises(ArtifactError, match="checksum"): TraceCodec().decode(modified)
    missing = tmp_path / "missing"
    TraceCodec().encode(trace, missing)
    next((missing / "arrays").glob("*.npy")).unlink()
    with pytest.raises(ArtifactError, match="missing"): TraceCodec().decode(missing)


def test_filesystem_store_never_overwrites_finalized_trace_or_started_manifest(tmp_path) -> None:
    trace = synthetic_trace()
    store = FilesystemRunArtifactStore(tmp_path)
    run_id = trace.episode_context.run_id
    store.create_run(run_id, {"run_id": str(run_id)})
    store.write_episode_trace(run_id, trace)
    loaded = store.read_episode_trace(run_id, trace.episode_context.task_id, trace.episode_context.episode_id)
    assert loaded.executed_actions[0].applied_action.tobytes() == trace.executed_actions[0].applied_action.tobytes()
    with pytest.raises(ArtifactError): store.write_episode_trace(run_id, trace)
    with pytest.raises(ArtifactError): store.create_run(run_id, {})
    run_path = tmp_path / safe_key(str(run_id))
    assert (run_path / "manifest.started.json").is_file()
    assert not (run_path / "manifest.completed.json").exists()


def test_safe_keys_allow_stable_slash_ids_but_reject_traversal() -> None:
    assert "/" not in safe_key("libero/spatial/0")
    with pytest.raises(ArtifactError): safe_key("../escape")


def test_configuration_and_run_manifests_are_immutable_and_distinguish_partial_runs(tmp_path) -> None:
    store = FilesystemRunArtifactStore(tmp_path)
    run_id = synthetic_trace().episode_context.run_id
    digest = "a" * 64
    snapshot = RunConfigurationSnapshot("kind: source\n", "kind: resolved\n", digest, digest)
    store.create_run(run_id, {"status": "started"})
    store.write_configuration(run_id, snapshot)
    run_path = store._run_path(run_id)
    assert (run_path / "manifest.started.json").is_file()
    assert not (run_path / "manifest.completed.json").exists()
    assert not (run_path / "manifest.failed.json").exists()
    with pytest.raises(ArtifactError, match="configuration already exists"):
        store.write_configuration(run_id, snapshot)
    assert (run_path / "source_config.yaml").read_text(encoding="utf-8") == snapshot.portable_source_yaml
    assert (run_path / "resolved_config.yaml").read_text(encoding="utf-8") == snapshot.resolved_config_yaml
    store.finalize_run(run_id, {"status": "completed"})
    assert (run_path / "manifest.completed.json").is_file()
    with pytest.raises(ArtifactError, match="already finalized"):
        store.mark_run_failed(run_id, {"status": "failed"})
