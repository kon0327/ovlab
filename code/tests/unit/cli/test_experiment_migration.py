"""Legacy migration copies and verifies before any original is quarantined."""

import json
from pathlib import Path

import pytest

from ovlab_benchctl import experiment_migration as migration_module
from ovlab_benchctl.experiment_migration import ExperimentMigration, MigrationError


def _legacy_tree(data: Path) -> None:
    train = data / "runs/train/train-1"
    train.mkdir(parents=True)
    (train / "training-plan.json").write_text("{}", encoding="utf-8")
    (train / "manifest.json").write_text('{"status":"completed"}', encoding="utf-8")
    (train / "result.json").write_text('{"status":"completed"}', encoding="utf-8")
    evaluate = data / "runs/evaluate/eval-1"
    evaluate.mkdir(parents=True)
    (evaluate / "manifest.started.json").write_text("{}", encoding="utf-8")
    (evaluate / "partial.bin").write_bytes(b"incomplete evidence")
    isolated = data / "exports/isolated/eval-1"
    isolated.mkdir(parents=True)
    (isolated / "metadata.json").write_text('{"export_type":"isolated"}', encoding="utf-8")
    old_derived = data / "exports/train/derived/train-1/profile/build-1"
    old_derived.mkdir(parents=True)
    (old_derived / "manifest.json").write_text("{}", encoding="utf-8")
    old_interactive = data / "exports/train/interactive/train-1/profile/build-1"
    old_interactive.mkdir(parents=True)
    (old_interactive / "index.html").write_text("legacy", encoding="utf-8")
    grouped = data / "exports/grouped/old-comparison"
    grouped.mkdir(parents=True)
    (grouped / "metadata.json").write_text('{"export_type":"grouped"}', encoding="utf-8")


def _migration(data: Path) -> ExperimentMigration:
    def verify_report(category, source, interactive):
        assert category == "train"
        assert source.is_dir()
        if not (interactive / "new-report.ok").is_file():
            raise MigrationError("new unified interactive report is not verified")

    return ExperimentMigration(data, verify_evaluation=lambda path: None,
                               verify_training=lambda path: None, verify_report=verify_report)


def test_migration_preview_apply_finalize_cleanup_preserves_and_verifies(tmp_path):
    data = tmp_path / "data"
    _legacy_tree(data)
    migration = _migration(data)
    preview = migration.preview()
    assert preview["run_count"] == 2
    assert preview["report_count"] == 1
    assert preview["copy_count"] == 4
    train = next(run for run in preview["runs"] if run["category"] == "train")
    assert train["destination_path"] == str(data / "runs/legacy-train-train-1/train/train-1")
    assert train["reportable"] is True
    assert train["source_export_paths"] == ["exports/train/derived/train-1", "exports/train/interactive/train-1"]
    assert (data / "runs/train/train-1").is_dir()  # Preview is read-only.

    prepared = migration.apply()
    assert prepared["state"] == "prepared"
    assert (data / "runs/train/train-1").is_dir()
    assert (data / "runs/legacy-train-train-1/train/train-1/training-plan.json").read_text() == "{}"
    spec = json.loads((data / "runs/legacy-train-train-1/.experiment/spec.json").read_text())
    assert spec["schema_version"] == "ovlab.experiment/v1"
    assert spec["kind"] == "workflow"
    assert spec["id"] == "legacy-train-train-1"
    assert json.loads((data / "exports/legacy-train-train-1/experiment.json").read_text())["status"] == "migrating"
    assert "Migrated train run" in (data / "exports/legacy-train-train-1/index.html").read_text()
    assert (data / "exports/legacy-evaluate-eval-1/evaluate/isolated/eval-1--legacy/metadata.json").is_file()
    assert (data / "exports/legacy-comparisons/evaluate/grouped/old-comparison/metadata.json").is_file()
    with pytest.raises(MigrationError, match="not verified"):
        migration.finalize()
    assert (data / "runs/train/train-1").is_dir()
    report_root = Path(train["interactive_root"])
    report_root.mkdir(parents=True)
    (report_root / "new-report.ok").touch()
    quarantined = migration.finalize()
    assert quarantined["state"] == "quarantined"
    assert not (data / "runs/train/train-1").exists()
    assert not (data / "exports/train/derived/train-1").exists()
    assert (data / ".experiment-migration-quarantine" / quarantined["migration_id"] / "runs/train/train-1").is_dir()
    complete = migration.cleanup()
    assert complete["state"] == "complete"
    assert not (data / ".experiment-migration-quarantine" / quarantined["migration_id"]).exists()
    assert (data / "runs/legacy-train-train-1/train/train-1").is_dir()
    assert json.loads((data / "exports/legacy-train-train-1/experiment.json").read_text())["status"] == "migrated"
    assert not (data / "runs/train").exists()
    assert not (data / "runs/evaluate").exists()
    assert not (data / "exports/train/derived").exists()
    assert not (data / "exports/grouped").exists()


def test_partial_evaluation_with_finalized_traces_requires_a_new_report(tmp_path):
    data = tmp_path / "data"
    _legacy_tree(data)
    partial = data / "runs/evaluate/eval-1/tasks/task-a/episodes/episode-1"
    partial.mkdir(parents=True)
    (partial / "trace.finalized.json").write_text("{}", encoding="utf-8")
    preview = _migration(data).preview()
    evaluation = next(run for run in preview["runs"] if run["category"] == "evaluate")
    assert evaluation["reportable"] is True
    assert evaluation["report_reason"] == "finalized partial evaluation traces"


def test_apply_rolls_back_new_copies_if_a_later_copy_fails(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _legacy_tree(data)
    migration = _migration(data)
    original = migration._copy_verified
    calls = 0

    def fail_second(source, destination, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MigrationError("simulated copy failure")
        return original(source, destination, expected)

    monkeypatch.setattr(migration, "_copy_verified", fail_second)
    with pytest.raises(MigrationError, match="simulated"):
        migration.apply()
    assert not migration.manifest_path.exists()
    assert (data / "runs/train/train-1").is_dir()
    assert (data / "runs/evaluate/eval-1").is_dir()
    assert not (data / "runs/legacy-train-train-1/train/train-1").exists()


def test_finalize_rejects_changed_source_and_forged_manifest_path(tmp_path):
    data = tmp_path / "data"
    _legacy_tree(data)
    migration = _migration(data)
    migration.apply()
    (data / "runs/train/train-1/result.json").write_text('{"status":"changed"}', encoding="utf-8")
    with pytest.raises(MigrationError, match="legacy source changed"):
        migration.finalize()
    assert (data / "runs/train/train-1").is_dir()
    document = json.loads(migration.manifest_path.read_text())
    document["entries"][0]["destination"] = "../outside"
    migration.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MigrationError, match="invalid migration destination"):
        migration.finalize()


def test_cleanup_keeps_quarantine_when_metadata_refresh_fails(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _legacy_tree(data)
    migration = _migration(data)
    prepared = migration.apply()
    train = next(run for run in prepared["runs"] if run["category"] == "train")
    report_root = Path(train["interactive_root"])
    report_root.mkdir(parents=True)
    (report_root / "new-report.ok").touch()
    quarantined = migration.finalize()
    quarantine = data / ".experiment-migration-quarantine" / quarantined["migration_id"]

    def fail_metadata(*args, **kwargs):
        raise MigrationError("simulated metadata write failure")

    original = migration._write_experiment_record
    monkeypatch.setattr(migration, "_write_experiment_record", fail_metadata)
    with pytest.raises(MigrationError, match="simulated metadata"):
        migration.cleanup()
    assert (quarantine / "runs/train/train-1").is_dir()
    monkeypatch.setattr(migration, "_write_experiment_record", original)
    assert migration.cleanup()["state"] == "complete"


def test_root_migration_restores_original_copy_ownership(tmp_path, monkeypatch):
    data = tmp_path / "data"
    source = data / "runs/train/run-1"
    source.mkdir(parents=True)
    (source / "private.bin").write_bytes(b"private")
    destination = data / "runs/legacy-train-run-1/train/run-1"
    migration = ExperimentMigration(data)
    expected = migration_module._inventory(source)
    observed = []
    monkeypatch.setattr(migration_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(migration_module.os, "chown", lambda path, uid, gid, **kwargs: observed.append((Path(path), uid, gid)))
    migration._copy_verified(source, destination, expected)
    assert len(observed) == 2
    assert any(path.name == "private.bin" for path, _, _ in observed)
    assert migration_module._inventory(destination) == expected
