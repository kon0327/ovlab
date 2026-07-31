from __future__ import annotations

import json
from pathlib import Path
import struct

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
    result = CheckpointBundleStore(tmp_path).finalize(context)
    checkpoint_id = result["checkpoint_id"]
    assert result["checkpoint"]["merge_status"] == "unmerged"
    verified = CheckpointBundleStore(tmp_path).verify(checkpoint_id)
    assert verified["status"] == "verified"
    assert inspect_safetensors(Path(result["checkpoint_path"]) / "weights-or-adapter" / "adapter_model.safetensors")[0]["dtype"] == "F32"


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
