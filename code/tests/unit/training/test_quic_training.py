from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from ovlab_benchctl.training_composition import load_training_profile
from ovlab_benchctl.training_errors import CheckpointBundleError, TrainingProfileError
from ovlab_benchctl.training_profiles import TrainingProfile, TrainingPlanner
from ovlab_benchctl.training_quic import validate_staged_adapter
from ovlab_benchctl.training_runs import CheckpointBundleStore, TrainingRunContext
from ovlab_benchctl.training_data import build_episode_manifest

ROOT = Path(__file__).resolve().parents[4]


def profile():
    value = deepcopy(load_training_profile(ROOT / "configs/experiments/training/openvla-libero10-lora-smoke.yaml").document)
    value["trainer"]["implementation"] = "openvla-quic"
    value["dataset"]["partition"] = {"seed": 42, "validation_fraction": .2, "action_normalization": "host"}
    value["training"].update(validation_batches=2, normalization={"unnorm_key": "libero_10"})
    value["training"]["peft"] = {"method": "quic", "placement": {"id": "custom"},
        "adapter": {"profile": "QP2"}, "targets": [{"names": ["language_model.model.layers.12.self_attn.o_proj"]}]}
    return value


def test_quic_profile_is_explicit_dense_adapter_only():
    value = profile()
    parsed = TrainingProfile.from_document(value)
    assert parsed.document["checkpointing"]["output_kind"] == "adapter"
    value["training"]["quantization"] = "4bit"
    with pytest.raises(TrainingProfileError, match="quantization"):
        TrainingProfile.from_document(value)


def test_quic_does_not_inherit_the_reference_lora_vram_requirement():
    value = profile()
    assert TrainingPlanner._estimated_vram(TrainingProfile.from_document(value)) == 23
    value["training"]["precision"] = "fp32"
    assert TrainingPlanner._estimated_vram(TrainingProfile.from_document(value)) == 45


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["dataset"].pop("partition"), "partition"),
    (lambda p: p["training"].pop("normalization"), "normalization"),
    (lambda p: p["training"]["peft"]["adapter"].update(operation="max"), "determinant"),
    (lambda p: p["training"]["peft"]["adapter"].update(orders=[2, 1]), "orders"),
    (lambda p: p["training"]["peft"]["adapter"].update(orthogonal=False), "orthogonal"),
    (lambda p: p["training"]["peft"]["targets"][0].update(pattern=".*"), "exactly one"),
    (lambda p: p["dataset"]["partition"].update(action_normalization="validation-q99"), "host or train-q99"),
    (lambda p: p["training"]["peft"].update(dropout=.1), "unknown"),
])
def test_invalid_quic_contract_fails_before_runtime_import(mutate, match):
    value = profile()
    mutate(value)
    with pytest.raises(TrainingProfileError, match=match):
        TrainingProfile.from_document(value)


def staged(tmp_path, *, fitted=False, names=("linear",), convolution=False):
    scientific = {"training": profile()["training"], "dataset": {"dataset_id": "dataset-fixture"},
        "base_checkpoint": {"resource_id": "base", "revision": "a" * 40, "aggregate_sha256": "b" * 64},
        "output": {"output_kind": "adapter"}}
    if fitted:
        scientific["profile"] = {"dataset": {"partition": {"action_normalization": "train-q99"}}}
    context = TrainingRunContext("quic-fixture", tmp_path / "runs/train/quic-fixture", {
        "scientific": scientific, "scientific_training_id": "training-plan-" + "1" * 32,
        "execution_plan_id": "training-execution-" + "2" * 32})
    weights = context.staging_checkpoint / "weights-or-adapter"
    weights.mkdir(parents=True)
    config = {"base_size": 2, "block_count": 1, "orders": [1], "share_across_blocks": False,
        "orthogonal": True, "orientation": "output", "construction_dtype": "float32"}
    manifest = [{"name": name, "config": deepcopy(config)} for name in names]
    if convolution:
        for entry in manifest:
            entry.update(module_type="Conv2d", in_features=3, out_features=32,
                         in_channels=3, out_channels=32, kernel_size=[14, 14],
                         stride=[14, 14], padding=[0, 0], dilation=[1, 1],
                         groups=1, padding_mode="zeros", bias=True)
    (context.root / "target-manifest.json").write_text(json.dumps({"targets": manifest}))
    normalization = {"unnorm_key": "libero_10", "action_statistics": {"q01": [-1] * 7, "q99": [1] * 7},
        "mode": "train-q99" if fitted else "host"}
    normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest()
    host_identity = {"checkpoint_sha256": "c" * 64}
    if fitted:
        host_identity["normalization_contract_sha256"] = normalization["sha256"]
    binding = {"format_version": 2 if convolution else 1, "host_identity": host_identity, "manifest": manifest}
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    header = json.dumps({"__metadata__": {"quic_binding_sha256": digest}, **{
        f"{name}.adapter.entries": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4*i, 4*(i+1)]}
        for i, name in enumerate(names)}}).encode()
    tensor = struct.pack("<Q", len(header)) + header + struct.pack("<" + "f" * len(names), *([.1] * len(names)))
    (weights / "adapter.safetensors").write_bytes(tensor)
    partition = build_episode_manifest([{"id": "episode-1", "group": "task", "frames": 2},
        {"id": "episode-2", "group": "task", "frames": 2}],
        {"seed": 42, "validation_fraction": .5, "action_normalization": "train-q99" if fitted else "host"},
        source_identity=scientific["dataset"])
    if fitted:
        from ovlab_benchctl.training_identity import identity
        partition["normalization"] = {"source": "train-q99", "fit_split": "train", "statistics": normalization["action_statistics"]}
        partition["manifest_id"] = identity("episode-partition", {key: value for key, value in partition.items() if key != "manifest_id"}, 32)
    (context.root / "data-partition.json").write_text(json.dumps(partition))
    (context.root / "normalization.json").write_text(json.dumps(normalization))
    payload = {**binding, "tensor_sha256": hashlib.sha256(tensor).hexdigest(),
        "metadata": {"base_checkpoint": scientific["base_checkpoint"], "peft": scientific["training"]["peft"],
            "dataset": scientific["dataset"], "normalization_contract": normalization,
            "split_manifest_id": partition["manifest_id"], "preprocessing": {"prompt_style": "modern", "center_crop": False}}}
    (weights / "adapter.json").write_text(json.dumps(payload))
    (context.root / "events.jsonl").write_text("")
    (context.root / "metrics.jsonl").write_text("")
    (context.root / "training-plan.json").write_text(json.dumps(context.plan))
    return context, weights, payload


def test_quic_bundle_publishes_with_its_own_loader_and_verifies(tmp_path):
    context, weights, _ = staged(tmp_path)
    result = CheckpointBundleStore(tmp_path).finalize(context)
    assert result["checkpoint"]["expected_loader"] == "ovlab-openvla-quic"
    assert result["checkpoint"]["merge_status"] == "unmerged"
    assert CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])["status"] == "verified"


def test_conv_bundle_finalizes_with_complete_geometry_and_version_two(tmp_path):
    context, weights, _ = staged(tmp_path, names=("vision_backbone.patch_embed.proj",), convolution=True)
    payload, _ = validate_staged_adapter(weights, context.plan["scientific"], run_root=context.root)
    assert payload["format_version"] == 2
    assert payload["manifest"][0]["kernel_size"] == [14, 14]
    result = CheckpointBundleStore(tmp_path).finalize(context)
    assert CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])["status"] == "verified"


@pytest.mark.parametrize("mutation", [{"groups": 2}, {"kernel_size": [14]},
                                     {"stride": [0, 14]}, {"padding_mode": "unknown"}])
def test_finalizer_rejects_invalid_conv_geometry_without_importing_torch(mutation):
    from ovlab_benchctl.training_quic import _validate_target_geometry
    entry = {"module_type": "Conv2d", "in_features": 3, "out_features": 32,
             "in_channels": 3, "out_channels": 32, "kernel_size": [14, 14],
             "stride": [14, 14], "padding": [0, 0], "dilation": [1, 1],
             "groups": 1, "padding_mode": "zeros", "config": {"block_count": 8}}
    entry.update(mutation)
    with pytest.raises(CheckpointBundleError, match="Conv2d target geometry"):
        _validate_target_geometry(entry, 2)


@pytest.mark.parametrize("tamper", [None, "changed", "missing", "duplicate"])
def test_finalizer_compares_complete_target_records_by_name_independently_of_order(tmp_path, tamper):
    context, weights, payload = staged(tmp_path, names=("layers.4.linear", "layers.12.linear"))
    recorded = deepcopy(list(reversed(payload["manifest"])))
    if tamper == "changed":
        recorded[0]["config"]["base_size"] = 3
    elif tamper == "missing":
        recorded.pop()
    elif tamper == "duplicate":
        recorded[1] = deepcopy(recorded[0])
    (context.root / "target-manifest.json").write_text(json.dumps({"targets": recorded}))
    if tamper:
        with pytest.raises(CheckpointBundleError, match="target manifest differs"):
            validate_staged_adapter(weights, context.plan["scientific"], run_root=context.root)
    else:
        result = CheckpointBundleStore(tmp_path).finalize(context)
        assert CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])["status"] == "verified"


def test_quic_finalizer_rejects_wrong_plan_and_tensor_tampering(tmp_path):
    context, weights, payload = staged(tmp_path)
    payload["metadata"]["base_checkpoint"] = {"aggregate_sha256": "d" * 64}
    (weights / "adapter.json").write_text(json.dumps(payload))
    with pytest.raises(CheckpointBundleError, match="base checkpoint"):
        validate_staged_adapter(weights, context.plan["scientific"])
    payload["metadata"]["base_checkpoint"] = context.plan["scientific"]["base_checkpoint"]
    (weights / "adapter.json").write_text(json.dumps(payload))
    with (weights / "adapter.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointBundleError, match="digest"):
        validate_staged_adapter(weights, context.plan["scientific"])


def test_quic_finalizer_rejects_missing_normalization_and_different_actual_split(tmp_path):
    context, weights, payload = staged(tmp_path)
    normalization = payload["metadata"].pop("normalization_contract")
    (weights / "adapter.json").write_text(json.dumps(payload))
    with pytest.raises(CheckpointBundleError, match="normalization"):
        validate_staged_adapter(weights, context.plan["scientific"], run_root=context.root)
    payload["metadata"]["normalization_contract"] = normalization
    payload["metadata"]["split_manifest_id"] = "different-split"
    (weights / "adapter.json").write_text(json.dumps(payload))
    with pytest.raises(CheckpointBundleError, match="split identity"):
        validate_staged_adapter(weights, context.plan["scientific"], run_root=context.root)


def test_fitted_normalization_bundle_finalizes_and_keeps_bound_contract(tmp_path):
    context, weights, payload = staged(tmp_path, fitted=True)
    validated, _ = validate_staged_adapter(weights, context.plan["scientific"], run_root=context.root)
    normalization = validated["metadata"]["normalization_contract"]
    assert normalization["mode"] == "train-q99"
    assert validated["host_identity"]["normalization_contract_sha256"] == normalization["sha256"]
    result = CheckpointBundleStore(tmp_path).finalize(context)
    assert result["checkpoint"]["adapter"]["metadata"]["normalization_contract"] == normalization
    assert CheckpointBundleStore(tmp_path).verify(result["checkpoint_id"])["status"] == "verified"


@pytest.mark.parametrize("tamper,match", [
    ("planned-mode", "normalization mode"),
    ("host-digest", "bound to the QuIC host"),
    ("fit-split", "train-only partition statistics"),
    ("partition-statistics", "train-only partition statistics"),
])
def test_fitted_finalizer_rejects_mode_host_binding_or_partition_statistics_drift(tmp_path, tamper, match):
    from ovlab_benchctl.training_identity import identity
    context, weights, payload = staged(tmp_path, fitted=True)
    scientific = context.plan["scientific"]
    if tamper == "planned-mode":
        scientific["profile"]["dataset"]["partition"]["action_normalization"] = "host"
    elif tamper == "host-digest":
        payload["host_identity"]["normalization_contract_sha256"] = "0" * 64
    else:
        partition_path = context.root / "data-partition.json"
        partition = json.loads(partition_path.read_text())
        if tamper == "fit-split":
            partition["normalization"]["fit_split"] = "validation"
        else:
            partition["normalization"]["statistics"]["q99"][0] = 1000000
        partition["manifest_id"] = identity("episode-partition", {key: value for key, value in partition.items() if key != "manifest_id"}, 32)
        partition_path.write_text(json.dumps(partition))
        payload["metadata"]["split_manifest_id"] = partition["manifest_id"]
    (weights / "adapter.json").write_text(json.dumps(payload))
    with pytest.raises(CheckpointBundleError, match=match):
        validate_staged_adapter(weights, scientific, run_root=context.root)
