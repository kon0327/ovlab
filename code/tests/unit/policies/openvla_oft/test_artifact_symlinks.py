from pathlib import Path

from ovlab_openvla_oft import OftFileIdentity, OpenVlaOftArtifact


def test_artifact_verification_accepts_huggingface_style_blob_symlinks(tmp_path: Path):
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / ("0" * 40)
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    required = {
        "config.json": b"config", "dataset_statistics.json": b"stats",
        "model.safetensors.index.json": b"index",
        "model-00001-of-00004.safetensors": b"1", "model-00002-of-00004.safetensors": b"2",
        "model-00003-of-00004.safetensors": b"3", "model-00004-of-00004.safetensors": b"4",
        "action_head--150000_checkpoint.pt": b"head",
        "proprio_projector--150000_checkpoint.pt": b"projector",
        "lora_adapter/adapter_config.json": b"adapter-config",
        "lora_adapter/adapter_model.safetensors": b"adapter",
    }
    import hashlib
    identities = []
    for index, (name, value) in enumerate(required.items()):
        blob = blobs / str(index)
        blob.write_bytes(value)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        identities.append(OftFileIdentity(name, len(value), hashlib.sha256(value).hexdigest()))
    identities = tuple(sorted(identities, key=lambda item: item.path))
    aggregate = hashlib.sha256("".join(item.manifest_line() for item in identities).encode()).hexdigest()
    method = {
        "family": "openvla_oft", "acronym_expansion": "optimized_fine_tuning",
        "backbone_adaptation": "lora", "artifact_form": "merged_backbone_with_auxiliary_components",
        "backbone_merge_status": "merged", "runtime_active_adapter": False,
        "parallel_decoding": True, "action_representation": "continuous", "objective": "l1_regression",
        "action_chunk_size": 8, "action_dimension": 7, "normalization": "bounds_q99",
        "image_inputs": 2, "proprioception_dimension": 8, "film": False, "diffusion": False,
        "quantization": "none", "adaptation_suite": "LIBERO-10",
        "dataset_identity": "libero_10_no_noops",
    }
    artifact = OpenVlaOftArtifact(
        "test", "moojink/openvla-7b-oft-finetuned-libero-10", "0" * 40,
        aggregate, identities, method, {"x": 1}, {"x": 1},
    )
    assert artifact.verify(snapshot)["resolved_snapshot_path"] == str(snapshot.resolve())
