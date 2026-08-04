from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ovlab_benchctl.data_management import DataManager, DataSafetyError
from ovlab_benchctl.cli import _confirm_data_operation
from ovlab_benchctl.run_references import run_hash


REPOSITORY = Path(__file__).resolve().parents[4]


def _tree(data: Path):
    run = data / "runs/run-1"
    run.mkdir(parents=True)
    (run / "manifest.completed.json").write_text("{}", encoding="utf-8")
    (run / "trace.bin").write_bytes(b"canonical")
    report = data / "derived/run-1/libero-task-default/build-1"
    report.mkdir(parents=True)
    (report / "manifest.json").write_text("{}", encoding="utf-8")
    training = data / "derived/training/training-1/system-performance/build-1"
    training.mkdir(parents=True)
    (training / "manifest.json").write_text("{}", encoding="utf-8")
    isolated = data / "exports/isolated/run-1"
    isolated.mkdir(parents=True)
    (isolated / "metadata.json").write_text(
        json.dumps({
            "schema_version": "ovlab.export-metadata/v2",
            "export_type": "isolated",
        }),
        encoding="utf-8",
    )
    grouped = data / "exports/grouped/study-1"
    grouped.mkdir(parents=True)
    (grouped / "metadata.json").write_text(
        json.dumps({
            "schema_version": "ovlab.export-metadata/v2",
            "export_type": "grouped",
        }),
        encoding="utf-8",
    )
    untouched = data / "datasets/keep-me"
    untouched.mkdir(parents=True)
    (untouched / "data.bin").write_bytes(b"dataset")
    return run, report.parents[1], training.parents[1], isolated, grouped, untouched


def _invoke(data: Path, *arguments):
    return subprocess.run(
        [str(REPOSITORY / "ovlab"), *arguments], cwd=REPOSITORY,
        env={
            **os.environ, "OVLAB_PYTHON": sys.executable,
            "OVLAB_DATA_ROOT": str(data),
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def test_data_list_compact_detail_and_training_report_identity(tmp_path):
    data = tmp_path / "data"
    _tree(data)
    compact = _invoke(data, "data", "list")
    assert compact.returncode == 0 and compact.stderr == ""
    assert f"run run-1 ({run_hash('run-1')}) completed {data / 'runs/run-1'}" in compact.stdout
    assert f"report run-1 generated {data / 'derived/run-1'}" in compact.stdout
    assert f"report training:training-1 generated {data / 'derived/training/training-1'}" in compact.stdout
    assert f"export isolated:run-1 generated {data / 'exports/isolated/run-1'}" in compact.stdout
    assert f"export grouped:study-1 generated {data / 'exports/grouped/study-1'}" in compact.stdout

    detailed = _invoke(data, "data", "list", "--detail")
    document = json.loads(detailed.stdout)
    assert detailed.returncode == 0
    assert document["schema_version"] == "ovlab.data-list/v1"
    assert all("size_bytes" in item and "file_count" in item for item in document["items"])
    run_item = next(item for item in document["items"] if item["kind"] == "run")
    assert run_item["run_hash"] == run_hash("run-1")


def test_run_report_and_isolated_export_accept_displayed_run_hash(tmp_path):
    data = tmp_path / "data"
    run, report, _, isolated, _, _ = _tree(data)
    alias = run_hash("run-1")

    run_preview = _invoke(data, "data", "delete", "--run", alias, "--dry-run", "--json")
    report_preview = _invoke(data, "data", "delete", "--report", alias, "--dry-run", "--json")
    export_preview = _invoke(
        data, "data", "delete", "--export", f"isolated:{alias}", "--dry-run", "--json",
    )

    assert run_preview.returncode == report_preview.returncode == export_preview.returncode == 0
    assert json.loads(run_preview.stdout)["result"]["targets"][0]["id"] == "run-1"
    assert json.loads(report_preview.stdout)["result"]["targets"][0]["id"] == "run-1"
    assert json.loads(export_preview.stdout)["result"]["targets"][0]["id"] == "isolated:run-1"
    assert run.is_dir() and report.is_dir() and isolated.is_dir()


def test_delete_requires_confirmation_and_dry_run_never_mutates(tmp_path):
    data = tmp_path / "data"
    run, *_ = _tree(data)
    preview = _invoke(data, "data", "delete", "--run", "run-1", "--dry-run", "--json")
    assert preview.returncode == 0 and run.is_dir()
    assert json.loads(preview.stdout)["result"]["status"] == "planned"

    refused = _invoke(data, "data", "delete", "--run", "run-1")
    assert refused.returncode == 2 and run.is_dir()
    assert "interactive terminal, --yes, or --dry-run" in refused.stderr

    deleted = _invoke(data, "data", "delete", "--run", "run-1", "--yes", "--json")
    assert deleted.returncode == 0 and not run.exists()
    assert json.loads(deleted.stdout)["result"]["target_count"] == 1
    assert (data / "derived/run-1").is_dir()


def test_export_selection_is_namespaced_and_does_not_remove_other_data(tmp_path):
    data = tmp_path / "data"
    run, _, _, isolated, grouped, _ = _tree(data)
    listing = _invoke(data, "data", "list", "--kind", "exports")
    assert listing.returncode == 0
    assert "export isolated:run-1 generated" in listing.stdout
    assert "export grouped:study-1 generated" in listing.stdout
    assert "run run-1" not in listing.stdout

    invalid = _invoke(data, "data", "delete", "--export", "study-1", "--yes")
    assert invalid.returncode == 7
    assert "isolated:NAME or grouped:NAME" in invalid.stderr

    deleted = _invoke(
        data, "data", "delete", "--export", "grouped:study-1", "--yes", "--json",
    )
    assert deleted.returncode == 0
    assert not grouped.exists()
    assert isolated.is_dir() and run.is_dir()


def test_archive_all_preserves_relative_layout_and_unrelated_model_data(tmp_path):
    data = tmp_path / "data"
    run, report, training_report, isolated, grouped, untouched = _tree(data)
    archived = _invoke(data, "data", "archive", "--all", "--yes", "--json")
    assert archived.returncode == 0 and archived.stderr == ""
    result = json.loads(archived.stdout)["result"]
    assert result["target_count"] == 5
    assert not run.exists() and not report.exists() and not training_report.exists()
    assert not isolated.exists() and not grouped.exists()
    assert (data / "archive/runs/run-1/trace.bin").read_bytes() == b"canonical"
    assert (data / "archive/derived/run-1/libero-task-default/build-1/manifest.json").is_file()
    assert (data / "archive/derived/training/training-1/system-performance/build-1/manifest.json").is_file()
    assert (data / "archive/exports/isolated/run-1/metadata.json").is_file()
    assert (data / "archive/exports/grouped/study-1/metadata.json").is_file()
    assert (data / "archive/manifests/run/run-1.json").is_file()
    assert (data / "archive/manifests/report/training--training-1.json").is_file()
    assert (data / "archive/manifests/export/isolated--run-1.json").is_file()
    assert (data / "archive/manifests/export/grouped--study-1.json").is_file()
    assert untouched.is_dir()

    listing = _invoke(data, "data", "list", "--archived", "--json")
    ids = {(item["kind"], item["id"]) for item in json.loads(listing.stdout)["result"]["items"]}
    assert ids == {
        ("run", "run-1"),
        ("report", "run-1"),
        ("report", "training:training-1"),
        ("export", "isolated:run-1"),
        ("export", "grouped:study-1"),
    }


def test_incomplete_artifacts_and_archive_collisions_are_rejected_before_mutation(tmp_path):
    data = tmp_path / "data"
    active = data / "runs/active-run"
    active.mkdir(parents=True)
    (active / "manifest.started.json").write_text("{}", encoding="utf-8")
    manager = DataManager(data, data / "runs", data / "derived")
    with pytest.raises(DataSafetyError, match="active-or-incomplete"):
        manager.archive(run_id="active-run")
    assert active.is_dir()

    _tree(data)
    collision = data / "archive/runs/run-1"
    collision.mkdir(parents=True)
    with pytest.raises(DataSafetyError, match="already exists"):
        manager.archive(run_id="run-1")
    assert (data / "runs/run-1").is_dir()


def test_force_delete_removes_incomplete_run_but_archive_remains_protected(tmp_path):
    data = tmp_path / "data"
    active = data / "runs/active-run"
    active.mkdir(parents=True)
    (active / "manifest.started.json").write_text("{}", encoding="utf-8")
    manager = DataManager(data, data / "runs", data / "derived")

    preview = manager.preview("delete", run_id="active-run", force=True)
    assert preview["force"] is True
    assert preview["targets"][0]["state"] == "active-or-incomplete"
    with pytest.raises(DataSafetyError, match="only for delete"):
        manager.preview("archive", run_id="active-run", force=True)

    result = manager.delete(run_id="active-run", force=True)
    assert result["force"] is True
    assert not active.exists()


def test_cli_force_delete_wires_preview_and_mutation_for_incomplete_run(tmp_path):
    data = tmp_path / "data"
    active = data / "runs/active-run"
    active.mkdir(parents=True)
    (active / "manifest.started.json").write_text("{}", encoding="utf-8")

    refused = _invoke(data, "data", "delete", "--run", "active-run", "--yes")
    assert refused.returncode == 7 and active.is_dir()
    preview = _invoke(
        data, "data", "delete", "--run", "active-run", "--force", "--dry-run", "--json",
    )
    assert preview.returncode == 0 and active.is_dir()
    assert json.loads(preview.stdout)["result"]["force"] is True
    deleted = _invoke(
        data, "data", "delete", "--run", "active-run", "--force", "--yes", "--json",
    )
    assert deleted.returncode == 0 and not active.exists()
    assert json.loads(deleted.stdout)["result"]["force"] is True


def test_aborted_manifest_has_distinct_list_state(tmp_path):
    data = tmp_path / "data"
    run = data / "runs/aborted-run"
    run.mkdir(parents=True)
    (run / "manifest.failed.json").write_text(
        '{"status":"aborted","failure_type":"KeyboardInterrupt"}', encoding="utf-8",
    )
    manager = DataManager(data, data / "runs", data / "derived")

    listing = manager.list(kind="runs")
    assert listing["items"][0]["state"] == "aborted"
    assert manager.preview("delete", run_id="aborted-run")["target_count"] == 1


def test_delete_permission_preflight_fails_before_any_selected_target_is_removed(tmp_path):
    data = tmp_path / "data"
    run, report, *_ = _tree(data)
    blocked = run / "blocked"
    blocked.mkdir()
    blocked.chmod(0o555)
    manager = DataManager(data, data / "runs", data / "derived")
    try:
        with pytest.raises(DataSafetyError, match="before any data was removed"):
            manager.delete(all_data=True)
        assert run.is_dir() and report.is_dir()
    finally:
        blocked.chmod(0o755)


def test_partial_report_file_blocks_the_complete_all_selection(tmp_path):
    data = tmp_path / "data"
    run, report, *_ = _tree(data)
    (report / "render.partial").write_text("in progress", encoding="utf-8")
    manager = DataManager(data, data / "runs", data / "derived")
    with pytest.raises(DataSafetyError, match="active-or-incomplete"):
        manager.delete(all_data=True)
    assert run.is_dir() and report.is_dir()


def test_invalid_export_metadata_blocks_the_complete_all_selection(tmp_path):
    data = tmp_path / "data"
    run, _, _, isolated, _, _ = _tree(data)
    (isolated / "metadata.json").write_text(
        '{"schema_version":"ovlab.export-metadata/v2","export_type":"grouped"}',
        encoding="utf-8",
    )
    manager = DataManager(data, data / "runs", data / "derived")
    with pytest.raises(DataSafetyError, match="incomplete"):
        manager.archive(all_data=True)
    assert run.is_dir() and isolated.is_dir()


def test_interactive_all_confirmation_requires_explicit_action_phrase(monkeypatch):
    class Input:
        def __init__(self, value): self.value = value
        def isatty(self): return True
        def readline(self): return self.value

    args = SimpleNamespace(dry_run=False, yes=False, json=False, all_data=True)
    preview = {"action": "delete", "target_count": 3}
    monkeypatch.setattr(sys, "stdin", Input("no\n"))
    assert _confirm_data_operation(args, preview) is False
    monkeypatch.setattr(sys, "stdin", Input("DELETE ALL\n"))
    assert _confirm_data_operation(args, preview) is True


def test_all_selection_rejects_symbolic_links(tmp_path):
    data = tmp_path / "data"
    (data / "runs").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (data / "runs/linked-run").symlink_to(outside, target_is_directory=True)
    manager = DataManager(data, data / "runs", data / "derived")
    with pytest.raises(DataSafetyError, match="symbolic link"):
        manager.preview("delete", all_data=True, force=True)


def test_managed_roots_must_not_overlap(tmp_path):
    data = tmp_path / "data"
    with pytest.raises(DataSafetyError, match="roots overlap"):
        DataManager(
            data,
            data / "runs",
            data / "derived",
            archive_root=data / "runs/archive",
        )
    with pytest.raises(DataSafetyError, match="roots overlap"):
        DataManager(
            data,
            data / "runs",
            data / "derived",
            exports_root=data / "derived/exports",
        )
