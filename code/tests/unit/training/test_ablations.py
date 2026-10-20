from copy import deepcopy
import json
from pathlib import Path

import pytest

from ovlab_benchctl.strict_yaml import dumps, load, loads
from ovlab_benchctl.training_ablations import TrainingAblation
from ovlab_benchctl.training_composition import load_training_profile
from ovlab_benchctl.training_errors import TrainingProfileError, TrainingRuntimeError
from ovlab_benchctl.training_data import build_episode_manifest, verify_episode_manifest
from ovlab_benchctl.training_identity import atomic_json

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/ablations/openvla-quic-placement-smoke.yaml"


def _partition():
    return {"seed": 42, "validation_fraction": .2, "max_episodes_per_group": 4, "frame_stride": 3, "max_frames_per_episode": 2}


def _records():
    return [{"id": f"tfds-{group}-{index}", "group": group, "frames": 10} for group in ("alpha", "beta") for index in range(8)]


def test_episode_partition_is_order_independent_stratified_and_disjoint():
    first = build_episode_manifest(_records(), _partition(), source_identity={"dataset_id": "dataset-a"})
    second = build_episode_manifest(list(reversed(_records())), _partition(), source_identity={"dataset_id": "dataset-a"})
    assert first == second
    verify_episode_manifest(first)
    train = {row["id"] for row in first["episodes"] if row["split"] == "train"}
    validation = {row["id"] for row in first["episodes"] if row["split"] == "validation"}
    assert train.isdisjoint(validation)
    assert first["counts"] == {"train": {"episodes": 6, "frames": 12}, "validation": {"episodes": 2, "frames": 4}}
    assert {row["group"] for row in first["episodes"] if row["split"] == "validation"} == {"alpha", "beta"}
    changed = build_episode_manifest(_records(), dict(_partition(), seed=43), source_identity={"dataset_id": "dataset-a"})
    assert changed["manifest_id"] != first["manifest_id"]


def test_partition_rejects_duplicate_records_small_groups_and_tampering():
    with pytest.raises(TrainingRuntimeError, match="duplicate"):
        build_episode_manifest(_records() + [_records()[0]], _partition(), source_identity={})
    with pytest.raises(TrainingRuntimeError, match="at least two"):
        build_episode_manifest(_records(), dict(_partition(), max_episodes_per_group=1), source_identity={})
    manifest = build_episode_manifest(_records(), _partition(), source_identity={})
    manifest["episodes"][0]["split"] = "other"
    with pytest.raises(TrainingRuntimeError, match="identity mismatch"):
        verify_episode_manifest(manifest)


def test_training_profile_inheritance_and_quic_yaml_roundtrip():
    profile = load_training_profile(ROOT / "configs/experiments/training/quic/libero10-a9-qp2.yaml", config_root=ROOT / "configs")
    assert profile.document["training"]["peft"]["placement"]["id"] == "A9"
    assert len(profile.document["training"]["peft"]["targets"]) == 2
    assert loads(dumps(profile.document)) == profile.document


def test_training_extends_cannot_escape_root_or_cycle(tmp_path):
    root = tmp_path / "configs"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("id: outside\n")
    child = root / "child.yaml"
    child.write_text("extends: ../outside.yaml\n")
    with pytest.raises(TrainingProfileError, match="escapes"):
        load_training_profile(child, config_root=root)
    child.write_text("extends: parent.yaml\n")
    (root / "parent.yaml").write_text("extends: child.yaml\n")
    with pytest.raises(TrainingProfileError, match="cycle"):
        load_training_profile(child, config_root=root)


def _suite_copy(tmp_path, mutate):
    root = tmp_path / "configs"
    root.mkdir()
    base = load_training_profile(ROOT / "configs/experiments/training/openvla-libero10-quic-smoke.yaml").as_dict()
    first = deepcopy(base)
    first["id"] = "first"
    first["training"]["peft"]["adapter"]["block_count"] = 4
    second = deepcopy(base)
    second["id"] = "second"
    mutate(second)
    for name, value in (("base", base), ("first", first), ("second", second)):
        (root / f"{name}.yaml").write_text(dumps(value))
    suite = {"schema_version": "ovlab.training-ablation/v1", "id": "test", "base_profile": "base.yaml", "factors": ["training.peft.adapter.block_count"], "variants": {"first": "first.yaml", "second": "second.yaml"}}
    (root / "suite.yaml").write_text(dumps(suite))
    return root


def test_ablation_rejects_optimizer_or_split_drift(tmp_path):
    root = _suite_copy(tmp_path, lambda document: document["training"].update(seed=5))
    with pytest.raises(TrainingProfileError, match="undeclared/shared fields: training.seed"):
        TrainingAblation(root / "suite.yaml", config_root=root)


def test_ablation_materializes_existing_profiles_without_model_imports(tmp_path):
    suite = TrainingAblation(SUITE, config_root=ROOT / "configs")
    first, second = suite.plan(), suite.plan()
    assert first["ablation_id"] == second["ablation_id"]
    result = suite.materialize(tmp_path / "study")
    for path in result["profiles"].values():
        load_training_profile(path)
    assert json.loads((tmp_path / "study/plan.json").read_text())["ablation_id"] == first["ablation_id"]
    with pytest.raises(FileExistsError):
        suite.materialize(tmp_path / "study")


def test_ablation_runner_keeps_failure_evidence_for_partition_drift(tmp_path):
    suite = TrainingAblation(SUITE, config_root=ROOT / "configs")
    class Deployment:
        count = 0
        def run(self, profile):
            self.count += 1
            child = tmp_path / f"run-{self.count}"
            child.mkdir()
            partition = build_episode_manifest(_records(), dict(_partition(), seed=self.count), source_identity={"dataset_id": "dataset-a"})
            atomic_json(child / "data-partition.json", partition)
            return {"run_id": child.name, "host_path": str(child), "result": {"status": "completed"}}
    with pytest.raises(TrainingRuntimeError, match="shared episode split"):
        suite.run(tmp_path / "study", Deployment(), plan=suite.plan())
    result = json.loads((tmp_path / "study/result.json").read_text())
    assert result["status"] == "failed"
    assert len(result["variants"]) == 1
    assert (tmp_path / "study/shared-data-partition.json").is_file()


def test_fit_override_changes_training_identity_without_mutating_profile_or_files(monkeypatch, tmp_path):
    from ovlab_benchctl.training_profiles import TrainingPlanner, with_fitted_action_normalization
    path = ROOT / "configs/experiments/training/openvla-libero10-quic-smoke.yaml"
    source = path.read_bytes()
    host_profile = load_training_profile(path)
    original = deepcopy(host_profile.document)
    fitted = with_fitted_action_normalization(host_profile, True)
    assert fitted.document["dataset"]["partition"]["action_normalization"] == "train-q99"
    assert host_profile.document == original
    assert path.read_bytes() == source
    assert with_fitted_action_normalization(host_profile, False) is host_profile
    assert with_fitted_action_normalization(fitted, True).document == fitted.document
    planner = TrainingPlanner(ROOT, tmp_path)
    monkeypatch.setattr(planner, "_model", lambda _profile: {
        "resource_id": "base", "repository": "repo/base", "revision": "a" * 40,
        "aggregate_sha256": "b" * 64, "source_kind": "fixture", "host_path": "/host/base",
    })
    monkeypatch.setattr(planner, "_dataset", lambda _profile, allow_dataset_download: {
        "dataset_id": "dataset-fixture", "raw_content_digest": "c" * 64,
        "prepared_content_digest": "d" * 64, "preparation": {"format": "openvla-rlds"}, "host_path": "/host/data",
    })
    host_plan, fit_plan = planner.plan(host_profile), planner.plan(fitted)
    assert fit_plan["scientific_training_id"] != host_plan["scientific_training_id"]
    assert fit_plan["scientific"]["dataset"]["partition"]["action_normalization"] == "train-q99"
    assert host_plan["scientific"]["dataset"]["partition"]["action_normalization"] == "host"
    assert host_profile.document == original
    assert path.read_bytes() == source


def test_fit_override_requires_explicit_partition():
    from ovlab_benchctl.training_profiles import with_fitted_action_normalization
    profile = load_training_profile(ROOT / "configs/experiments/training/openvla-libero10-lora-smoke.yaml")
    with pytest.raises(TrainingProfileError, match="requires dataset.partition"):
        with_fitted_action_normalization(profile, True)


def test_fit_ablation_override_is_shared_materialized_and_separate_from_host_plan(tmp_path):
    source_paths = [SUITE, ROOT / "configs/experiments/training/openvla-libero10-quic-smoke.yaml", *(ROOT / "configs/experiments/training/quic").glob("*.yaml")]
    source_bytes = {path: path.read_bytes() for path in source_paths}
    host = TrainingAblation(SUITE, config_root=ROOT / "configs")
    fitted = TrainingAblation(SUITE, config_root=ROOT / "configs", fit_action_normalization=True)
    host_plan, fitted_plan = host.plan(), fitted.plan()
    assert host_plan["ablation_id"] != fitted_plan["ablation_id"]
    assert host_plan["scientific"]["partition_contract_id"] != fitted_plan["scientific"]["partition_contract_id"]
    assert host.common["dataset"]["partition"]["action_normalization"] == "host"
    assert fitted.common["dataset"]["partition"]["action_normalization"] == "train-q99"
    materialized = fitted.materialize(tmp_path / "fitted-study", plan=fitted_plan)
    for variant, path in materialized["profiles"].items():
        profile = load_training_profile(path)
        assert profile.document["dataset"]["partition"]["action_normalization"] == "train-q99"
        assert profile.document == fitted_plan["scientific"]["variants"][variant]
    persisted_before = (tmp_path / "fitted-study/plan.json").read_bytes()
    assert host.plan() == host_plan
    assert fitted.plan() == fitted_plan
    assert (tmp_path / "fitted-study/plan.json").read_bytes() == persisted_before
    assert all(path.read_bytes() == contents for path, contents in source_bytes.items())
