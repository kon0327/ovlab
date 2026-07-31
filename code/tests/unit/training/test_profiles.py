from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ovlab_benchctl.strict_yaml import load
from ovlab_benchctl.training_errors import TrainingProfileError, TrainingResourceError
from ovlab_benchctl.training_profiles import TrainingProfile, TrainingPlanner


ROOT = Path(__file__).resolve().parents[4]


def _lora():
    return load(ROOT / "configs/training/openvla-libero10-lora-smoke.yaml")


def _full():
    return load(ROOT / "configs/training/openvla-full-reference.yaml")


def test_lora_and_full_profiles_are_strict_and_deterministic():
    first = TrainingProfile.from_document(_lora())
    second = TrainingProfile.from_document(_lora())
    full = TrainingProfile.from_document(_full())
    assert first == second
    assert first.document["training"]["mode"] == "peft"
    assert full.document["training"]["mode"] == "full"
    assert full.document["checkpointing"]["output_kind"] == "full"


def test_unknown_field_and_quic_settings_are_rejected():
    value = _lora()
    value["training"]["shell_command"] = "python arbitrary.py"
    with pytest.raises(TrainingProfileError, match="unknown fields"):
        TrainingProfile.from_document(value)
    value = _lora()
    value["training"]["peft"]["method"] = "quic"
    with pytest.raises(TrainingProfileError, match="Gate J"):
        TrainingProfile.from_document(value)


def test_qlora_and_wrong_target_modules_are_rejected():
    value = _lora()
    value["training"]["quantization"] = "4bit"
    with pytest.raises(TrainingProfileError, match="QLoRA"):
        TrainingProfile.from_document(value)
    value = _lora()
    value["training"]["peft"]["target_modules"] = ["q_proj"]
    with pytest.raises(TrainingProfileError, match="all-linear"):
        TrainingProfile.from_document(value)


def test_profile_rejects_absolute_scientific_paths():
    value = _lora()
    value["model"]["base_checkpoint"] = "/tmp/model"
    with pytest.raises(TrainingProfileError, match="absolute"):
        TrainingProfile.from_document(value)


def test_resource_preflight_rejects_reference_lora_on_24_gib(monkeypatch, tmp_path):
    profile = TrainingProfile.from_document(_lora())
    planner = TrainingPlanner(ROOT, tmp_path)
    monkeypatch.setattr(planner, "_model", lambda _profile: {
        "resource_id": "base", "repository": "repo/base", "revision": "a" * 40,
        "aggregate_sha256": "b" * 64, "source_kind": "test", "host_path": "/host/base",
    })
    monkeypatch.setattr(planner, "_dataset", lambda _profile, allow_dataset_download: {
        "dataset_id": "dataset-abc", "raw_content_digest": "c" * 64,
        "prepared_content_digest": "d" * 64,
        "preparation": {"format": "openvla-rlds"}, "host_path": "/host/dataset",
    })
    with pytest.raises(TrainingResourceError, match="detected device memory 24.0 GiB"):
        planner.plan(profile, available_gpu_count=1, available_vram_gib=24.0)


def test_plan_id_is_stable_and_machine_paths_only_change_execution(monkeypatch, tmp_path):
    profile = TrainingProfile.from_document(_lora())
    planner = TrainingPlanner(ROOT, tmp_path)
    model = {
        "resource_id": "base", "repository": "repo/base", "revision": "a" * 40,
        "aggregate_sha256": "b" * 64, "source_kind": "test", "host_path": "/one/base",
    }
    dataset = {
        "dataset_id": "dataset-abc", "raw_content_digest": "c" * 64,
        "prepared_content_digest": "d" * 64,
        "preparation": {"format": "openvla-rlds"}, "host_path": "/one/dataset",
    }
    monkeypatch.setattr(planner, "_model", lambda _profile: dict(model))
    monkeypatch.setattr(planner, "_dataset", lambda _profile, allow_dataset_download: dict(dataset))
    first = planner.plan(profile, available_gpu_count=1, available_vram_gib=32.0, image_identity="image-a")
    model["host_path"] = "/two/base"
    dataset["host_path"] = "/two/dataset"
    second = planner.plan(profile, available_gpu_count=1, available_vram_gib=32.0, image_identity="image-b")
    assert first["scientific_training_id"] == second["scientific_training_id"]
    assert first["execution_plan_id"] != second["execution_plan_id"]
