from __future__ import annotations

import json
import errno
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import struct
import shutil
import threading

import pytest

from ovlab_benchctl import checkpointing
from ovlab_benchctl.checkpointing import (
    CheckpointResolutionError,
    ResolvedCheckpoint,
    checkpoint_spec_by_id,
    resolve_finalized_training_checkpoint,
)
from ovlab_benchctl.training_errors import CheckpointBundleError, TrainingRuntimeError
from ovlab_benchctl.training_runs import (
    CheckpointBundleStore,
    TrainingRunContext,
    TrainingRunStore,
    inspect_safetensors,
)


def _safetensors(path: Path, values=(1.0, 2.0)):
    payload = struct.pack("<" + "f" * len(values), *values)
    header = json.dumps({
        "base_model.model.layers.0.q_proj.lora_A.weight": {
            "dtype": "F32", "shape": [1, len(values)], "data_offsets": [0, len(payload)],
        }
    }, separators=(",", ":")).encode()
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


ROOT = Path(__file__).resolve().parents[4]


def _context(tmp_path, *, rank=2, base=None):
    plan = {
        "scientific_training_id": "training-plan-" + "a" * 32,
        "execution_plan_id": "training-execution-" + "b" * 32,
        "scientific": {
            "output": {"output_kind": "adapter"},
            "training": {"mode": "peft", "peft": {"method": "lora", "rank": rank, "alpha": 2, "target_modules": ["all-linear"]}},
            "base_checkpoint": base or {"resource_id": "base", "revision": "c" * 40, "aggregate_sha256": "d" * 64},
        },
    }
    run = tmp_path / "training-runs" / "run-1"
    weights = run / "staging-checkpoints" / "final" / "weights-or-adapter"
    weights.mkdir(parents=True)
    (weights / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 2, "lora_alpha": 2,
        "target_modules": ["all-linear"], "base_model_name_or_path": "/checkpoints/base",
    }), encoding="utf-8")
    _safetensors(weights / "adapter_model.safetensors")
    (run / "events.jsonl").write_text("", encoding="utf-8")
    (run / "metrics.jsonl").write_text("", encoding="utf-8")
    (run / "training-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return TrainingRunContext("run-1", run, plan)


def test_adapter_bundle_finalizes_unmerged_and_verifies(tmp_path):
    context = _context(tmp_path)
    (context.root / "result.json").write_text(json.dumps({
        "schema_version": "ovlab.training-result/v1", "status": "checkpointing",
        "total_parameter_count": 1000, "trainable_parameter_count": 100,
        "frozen_parameter_count": 900, "adapter_parameter_count": 100,
        "trainable_adapter_parameter_count": 100,
        "performance": {"schema_version": "ovlab.performance-summary/v1"},
    }), encoding="utf-8")
    result = CheckpointBundleStore(tmp_path).finalize(context)
    checkpoint_id = result["checkpoint_id"]
    assert Path(result["checkpoint_path"]) == tmp_path / "checkpoints" / "local" / checkpoint_id
    assert result["checkpoint"]["merge_status"] == "unmerged"
    verified = CheckpointBundleStore(tmp_path).verify(checkpoint_id)
    assert verified["status"] == "verified"
    assert inspect_safetensors(Path(result["checkpoint_path"]) / "weights-or-adapter" / "adapter_model.safetensors")[0]["dtype"] == "F32"
    finalized_result = json.loads((context.root / "result.json").read_text(encoding="utf-8"))
    assert finalized_result["performance"]["schema_version"] == "ovlab.performance-summary/v1"
    assert finalized_result["adapter_parameter_count"] == 100
    assert result["checkpoint"]["parameter_counts"]["adapter"] == 100


@pytest.mark.parametrize("link_error", [errno.EXDEV, errno.EACCES])
def test_finalizer_copies_across_bind_mounts_and_propagates_other_link_errors(tmp_path, monkeypatch, link_error):
    context = _context(tmp_path)
    private_tensor = context.staging_checkpoint / "weights-or-adapter/adapter_model.safetensors"
    private_tensor.chmod(0o600)
    original_link = os.link

    def link(source, target, *args, **kwargs):
        if Path(source).is_relative_to(context.staging_checkpoint):
            raise OSError(link_error, "simulated bind-mount link failure")
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)
    if link_error != errno.EXDEV:
        with pytest.raises(OSError) as failure:
            CheckpointBundleStore(tmp_path).finalize(context)
        assert failure.value.errno == link_error
        assert not list((tmp_path / "checkpoints/local").glob("checkpoint-*"))
        return
    result = CheckpointBundleStore(tmp_path).finalize(context)
    assert CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])["status"] == "verified"
    source = context.staging_checkpoint / "weights-or-adapter/adapter_model.safetensors"
    published = Path(result["checkpoint_path"]) / "weights-or-adapter/adapter_model.safetensors"
    assert published.read_bytes() == source.read_bytes()
    assert published.stat().st_ino != source.stat().st_ino
    assert source.stat().st_mode & 0o040  # Host group can verify worker-private staging files.
    assert TrainingRunStore(tmp_path, training_runs_root=context.root.parent).verify(context.run_id)["status"] == "verified"


def test_reusing_checkpoint_finalizes_new_training_run_and_preserves_original(tmp_path):
    first_context = _context(tmp_path)
    first = CheckpointBundleStore(tmp_path).finalize(first_context)
    first_checkpoint_document = (Path(first["checkpoint_path"]) / "checkpoint.json").read_bytes()
    second_root = tmp_path / "training-runs" / "run-2"
    second_root.mkdir()
    shutil.copytree(first_context.staging_checkpoint, second_root / "staging-checkpoints/final")
    for name in ("events.jsonl", "metrics.jsonl", "training-plan.json"):
        shutil.copyfile(first_context.root / name, second_root / name)
    (second_root / "result.json").write_text(json.dumps({"status": "checkpointing", "optimizer_steps": 3}))
    second_context = TrainingRunContext("run-2", second_root, first_context.plan)

    second = CheckpointBundleStore(tmp_path).finalize(second_context)

    assert second["reused"] is True
    assert second["checkpoint_id"] == first["checkpoint_id"]
    assert second["run_id"] == "run-2"
    assert second["status"] == "completed"
    assert second["optimizer_steps"] == 3
    assert json.loads((second_root / "result.json").read_text())["status"] == "completed"
    assert TrainingRunStore(tmp_path, training_runs_root=second_root.parent).verify("run-2")["status"] == "verified"
    assert (Path(first["checkpoint_path"]) / "checkpoint.json").read_bytes() == first_checkpoint_document
    repeated = CheckpointBundleStore(tmp_path).finalize(second_context)
    assert repeated["reused"] is True
    assert repeated["run_id"] == "run-2"


def test_parallel_finalizers_publish_identical_checkpoint_once(tmp_path, monkeypatch):
    first = _context(tmp_path)
    second_root = tmp_path / "training-runs" / "run-2"
    second_root.mkdir()
    shutil.copytree(first.staging_checkpoint, second_root / "staging-checkpoints/final")
    for name in ("events.jsonl", "metrics.jsonl", "training-plan.json"):
        shutil.copyfile(first.root / name, second_root / name)
    second = TrainingRunContext("run-2", second_root, first.plan)
    barrier = threading.Barrier(2, timeout=10)
    original = CheckpointBundleStore._publication_lock

    @contextmanager
    def synchronized(store, checkpoint_id):
        barrier.wait()
        with original(store, checkpoint_id):
            yield

    monkeypatch.setattr(CheckpointBundleStore, "_publication_lock", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda context: CheckpointBundleStore(tmp_path).finalize(context), (first, second)))
    assert results[0]["checkpoint_id"] == results[1]["checkpoint_id"]
    assert sorted(result["reused"] for result in results) == [False, True]
    assert CheckpointBundleStore(tmp_path).verify(results[0]["checkpoint_id"])["status"] == "verified"
    assert all(TrainingRunStore(tmp_path, training_runs_root=context.root.parent).verify(context.run_id)["status"] == "verified"
               for context in (first, second))
    assert not list((tmp_path / "checkpoints/local").glob("*.staging"))


def test_incompatible_adapter_configuration_is_rejected(tmp_path):
    context = _context(tmp_path, rank=4)
    with pytest.raises(CheckpointBundleError, match="does not match"):
        CheckpointBundleStore(tmp_path).finalize(context)


def test_nonfinite_adapter_tensor_is_rejected(tmp_path):
    context = _context(tmp_path)
    _safetensors(context.staging_checkpoint / "weights-or-adapter" / "adapter_model.safetensors", (float("nan"), 1.0))
    with pytest.raises(CheckpointBundleError, match="non-finite"):
        CheckpointBundleStore(tmp_path).finalize(context)


def test_checkpoint_tampering_is_detected(tmp_path):
    context = _context(tmp_path)
    result = CheckpointBundleStore(tmp_path).finalize(context)
    target = Path(result["checkpoint_path"]) / "weights-or-adapter" / "adapter_model.safetensors"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(CheckpointBundleError, match="integrity failure"):
        CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])


def test_failed_and_interrupted_runs_never_verify_as_completed(tmp_path):
    store = TrainingRunStore(tmp_path)
    profile = {"id": "fixture"}
    plan = {"scientific_training_id": "training-plan-abc", "execution_plan_id": "execution-abc"}
    context = store.create(profile, profile, plan, {})
    store.fail(context, RuntimeError("first raw failure"), interrupted=True)
    assert store.inspect(context.run_id)["result"]["status"] == "interrupted"
    with pytest.raises(TrainingRuntimeError, match="not finalized"):
        store.verify(context.run_id)


def test_explicit_deployment_handoff_resolves_verified_unmerged_adapter_and_base(monkeypatch, tmp_path):
    spec = checkpoint_spec_by_id(ROOT, "openvla-7b")
    base_identity = {
        "resource_id": spec.resource_id,
        "repository": spec.repo_id,
        "revision": spec.revision,
        "aggregate_sha256": spec.expected_sha256,
    }
    result = CheckpointBundleStore(tmp_path).finalize(_context(tmp_path, base=base_identity))

    def resolve_base(_self, selected, *, local_path=None, offline=False):
        assert selected == spec
        assert offline is True
        return ResolvedCheckpoint(selected, tmp_path / "base", "/checkpoints/resolved/openvla-7b", "fixture", len(spec.files), False)

    monkeypatch.setattr(checkpointing.CheckpointResolver, "resolve", resolve_base)
    handoff = resolve_finalized_training_checkpoint(ROOT, tmp_path, result["checkpoint_id"], global_cache=tmp_path / "hf")
    assert handoff.kind == "peft_adapter"
    assert handoff.merge_status == "unmerged"
    assert handoff.base.spec.resource_id == "openvla-7b"
    assert handoff.as_dict()["dataset_required"] is False
    assert handoff.as_dict()["training_runtime_started"] is False


def test_deployment_handoff_rejects_alias_incomplete_and_incompatible_artifacts(monkeypatch, tmp_path):
    with pytest.raises(CheckpointResolutionError, match="aliases are not accepted"):
        resolve_finalized_training_checkpoint(ROOT, tmp_path, "latest")
    with pytest.raises(CheckpointResolutionError, match="not finalized"):
        resolve_finalized_training_checkpoint(ROOT, tmp_path, "checkpoint-" + "a" * 32)

    result = CheckpointBundleStore(tmp_path).finalize(_context(tmp_path))
    with pytest.raises(CheckpointResolutionError, match="registry resource"):
        resolve_finalized_training_checkpoint(ROOT, tmp_path, result["checkpoint_id"])


def test_legacy_training_checkpoint_remains_listable_and_verifiable(tmp_path):
    context = _context(tmp_path)
    store = CheckpointBundleStore(tmp_path)
    result = store.finalize(context)
    checkpoint_id = result["checkpoint_id"]
    current = Path(result["checkpoint_path"])
    legacy = tmp_path / "checkpoints" / checkpoint_id
    current.chmod(0o755)
    current.rename(legacy)
    legacy.chmod(0o555)

    assert store.inspect(checkpoint_id)["host_path"] == str(legacy)
    assert store.verify(checkpoint_id)["status"] == "verified"
    assert [row["checkpoint_id"] for row in store.list()] == [checkpoint_id]


def test_checkpoint_cannot_exist_in_both_local_and_legacy_locations(tmp_path):
    store = CheckpointBundleStore(tmp_path)
    result = store.finalize(_context(tmp_path))
    checkpoint_id = result["checkpoint_id"]
    shutil.copytree(Path(result["checkpoint_path"]), tmp_path / "checkpoints" / checkpoint_id)
    with pytest.raises(CheckpointBundleError, match="both local and legacy"):
        store.inspect(checkpoint_id)
    with pytest.raises(CheckpointBundleError, match="both local and legacy"):
        store.list()


def test_quic_action_head_bundle_verifies_both_tensor_files(tmp_path):
    from ovlab_benchctl.training_runs import CHECKPOINT_MANIFEST_SCHEMA
    from ovlab_benchctl.training_identity import inventory

    checkpoint_id = "checkpoint-" + "a" * 32
    root = tmp_path / "checkpoints/local" / checkpoint_id
    weights = root / "weights-or-adapter"
    weights.mkdir(parents=True)
    _safetensors(weights / "adapter.safetensors")
    _safetensors(weights / "action_head.safetensors")
    (root / "checkpoint.json").write_text(json.dumps({
        "kind": "peft_adapter", "expected_loader": "ovlab-openvla-quic",
        "adapter": {"metadata": {"action_head": {"config": {"horizon": 8}}}},
    }))
    files, _, size = inventory(root)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA, "state": "finalized",
        "checkpoint_id": checkpoint_id, "files": files, "total_size": size,
    }))
    store = CheckpointBundleStore(tmp_path)
    assert store.verify(checkpoint_id)["status"] == "verified"
    (weights / "action_head.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointBundleError, match="integrity failure"):
        store.verify(checkpoint_id)


def test_checkpoint_migration_dry_run_and_apply(tmp_path):
    import subprocess
    import sys

    store = CheckpointBundleStore(tmp_path)
    finalized = store.finalize(_context(tmp_path))
    checkpoint_id = finalized["checkpoint_id"]
    local = Path(finalized["checkpoint_path"])
    legacy = tmp_path / "checkpoints" / checkpoint_id
    local.chmod(0o755)
    local.rename(legacy)
    script = ROOT / "deploy/scripts/migrate-checkpoint-layout.py"
    base = [sys.executable, str(script), "--model-data-root", str(tmp_path)]
    planned = subprocess.run(base, text=True, capture_output=True, check=False)
    assert planned.returncode == 0 and "Dry run" in planned.stdout
    assert legacy.is_dir() and not local.exists()
    moved = subprocess.run([*base, "--apply"], text=True, capture_output=True, check=False)
    assert moved.returncode == 0, moved.stderr
    assert local.is_dir() and not legacy.exists()
    assert (tmp_path / "checkpoints/others").is_dir()
    assert store.verify(checkpoint_id)["status"] == "verified"


@pytest.mark.parametrize("export_format", ["bundle", "tar"])
def test_checkpoint_export_round_trip_and_refuses_overwrite(tmp_path, export_format):
    import tarfile
    from ovlab_benchctl.checkpoint_export import CheckpointExportError, export_checkpoint

    finalized = CheckpointBundleStore(tmp_path).finalize(_context(tmp_path))
    checkpoint_id = finalized["checkpoint_id"]
    target = tmp_path / ("export.tar" if export_format == "tar" else "export-bundle")
    result = export_checkpoint(tmp_path, checkpoint_id, format=export_format, output=target)
    assert result["integrity"] == "verified" and result["format"] == export_format
    if export_format == "bundle":
        assert (target / "checkpoint.json").read_bytes() == (Path(finalized["checkpoint_path"]) / "checkpoint.json").read_bytes()
    else:
        with tarfile.open(target) as archive:
            assert archive.extractfile(f"{checkpoint_id}/checkpoint.json").read() == (Path(finalized["checkpoint_path"]) / "checkpoint.json").read_bytes()
    with pytest.raises(CheckpointExportError, match="already exists"):
        export_checkpoint(tmp_path, checkpoint_id, format=export_format, output=target)
    assert CheckpointBundleStore(tmp_path).verify(checkpoint_id)["status"] == "verified"


def test_checkpoint_export_rejects_tampered_source_and_unsupported_onnx(tmp_path):
    from ovlab_benchctl.checkpoint_export import CheckpointExportError, export_checkpoint

    finalized = CheckpointBundleStore(tmp_path).finalize(_context(tmp_path))
    checkpoint_id = finalized["checkpoint_id"]
    target = tmp_path / "output"
    with pytest.raises(CheckpointExportError, match="ONNX export is not yet implemented"):
        export_checkpoint(tmp_path, checkpoint_id, format="onnx", output=target)
    assert not target.exists()
    weight = Path(finalized["checkpoint_path"]) / "weights-or-adapter/adapter_model.safetensors"
    weight.chmod(0o644)
    weight.write_bytes(b"corrupted")
    with pytest.raises(CheckpointBundleError, match="integrity failure"):
        export_checkpoint(tmp_path, checkpoint_id, format="bundle", output=target)
    assert not target.exists()


def test_checkpoint_export_cli_produces_verified_bundle(tmp_path):
    import subprocess
    import sys

    finalized = CheckpointBundleStore(tmp_path).finalize(_context(tmp_path))
    checkpoint_id = finalized["checkpoint_id"]
    output = tmp_path / "exported"
    env = {**os.environ, "OVLAB_MODEL_DATA_ROOT": str(tmp_path), "OVLAB_PYTHON": sys.executable}
    completed = subprocess.run(
        [str(ROOT / "ovlab"), "checkpoint", "export", checkpoint_id, "--format", "bundle", "--output", str(output), "--json"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)["result"]
    assert result["checkpoint_id"] == checkpoint_id and result["integrity"] == "verified"
    assert (output / "manifest.json").is_file()
