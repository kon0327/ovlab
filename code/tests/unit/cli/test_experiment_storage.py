"""Experiment trees keep runs and exports isolated at arbitrary node depth."""

import json

import pytest

from ovlab_core.experiment_scope import ExperimentScope
from ovlab_benchctl.data_management import DataManager, DataSafetyError
from ovlab_benchctl.run_references import RunReferenceAmbiguousError, RunReferenceError, resolve_run_reference_tree
from ovlab_benchctl.training_deployment import TrainingDeployment
from ovlab_benchctl.training_runs import TrainingRunStore


def test_scope_is_portable_and_recovers_location_from_canonical_run(tmp_path):
    scope = ExperimentScope.from_environment({"OVLAB_EXPERIMENT_ID": "study-1", "OVLAB_EXPERIMENT_NODE": "model-a/task-2/ablation-1"})
    assert scope.runs_root(tmp_path / "runs", "train") == tmp_path / "runs/study-1/train/model-a/task-2/ablation-1"
    assert scope.exports_root(tmp_path / "exports", "evaluate") == tmp_path / "exports/study-1/evaluate/model-a/task-2/ablation-1"
    run = scope.runs_root(tmp_path / "runs", "evaluate") / "run-1"
    assert ExperimentScope.from_run_path(tmp_path / "runs", run, "evaluate") == scope
    assert ExperimentScope.from_environment({}) is None
    with pytest.raises(ValueError, match="requires"):
        ExperimentScope.from_environment({"OVLAB_EXPERIMENT_NODE": "model-a"})
    for unsafe in ("../other", "model//task", "interactive/task", "model/.."):
        with pytest.raises(ValueError):
            ExperimentScope.from_environment({"OVLAB_EXPERIMENT_ID": "study-1", "OVLAB_EXPERIMENT_NODE": unsafe})


def test_nested_run_lookup_rejects_ambiguous_ids(tmp_path):
    runs = tmp_path / "runs"
    for experiment in ("study-a", "study-b"):
        run = runs / experiment / "evaluate" / "model" / "task" / "run-1"
        run.mkdir(parents=True)
        (run / "manifest.started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RunReferenceAmbiguousError, match="ambiguous"):
        resolve_run_reference_tree(runs, "run-1", category="evaluate")
    assert resolve_run_reference_tree(runs / "study-a", "run-1", category="evaluate").path == (
        runs / "study-a/evaluate/model/task/run-1"
    )
    selected = runs / "study-b/evaluate/model/task/run-1"
    assert resolve_run_reference_tree(runs, str(selected), category="evaluate").path == selected
    with pytest.raises(Exception, match="outside"):
        resolve_run_reference_tree(runs, str(tmp_path / "outside"), category="evaluate")
    link = runs / "linked-run"
    link.symlink_to(selected, target_is_directory=True)
    with pytest.raises(RunReferenceError, match="symbolic link"):
        resolve_run_reference_tree(runs, str(link), category="evaluate")


def test_training_store_writes_scoped_run_and_global_lookup_remains_available(tmp_path):
    scope = ExperimentScope("study-a", ("model-a", "task-a", "ablation-1"))
    store = TrainingRunStore(tmp_path, scope=scope)
    context = store.create({"id": "profile"}, {"id": "profile"},
                           {"scientific_training_id": "training-plan-abc"}, {})
    assert context.root.parent == tmp_path / "runs/study-a/train/model-a/task-a/ablation-1"
    assert context.root.stat().st_mode & 0o7777 == 0o2770
    provenance = json.loads((context.root / "provenance.json").read_text())
    assert provenance["experiment_scope"] == {"experiment_id": "study-a", "node_path": "model-a/task-a/ablation-1"}
    assert TrainingRunStore(tmp_path).inspect(context.run_id)["host_path"] == str(context.root)


def test_training_run_ids_are_distinct_across_experiment_nodes(tmp_path):
    plan = {"scientific_training_id": "training-plan-same"}
    first = TrainingRunStore(tmp_path, scope=ExperimentScope("study-a", ("model-a",))).create(
        {"id": "profile"}, {"id": "profile"}, plan, {},
    )
    second = TrainingRunStore(tmp_path, scope=ExperimentScope("study-a", ("model-b",))).create(
        {"id": "profile"}, {"id": "profile"}, plan, {},
    )
    assert first.run_id != second.run_id
    assert first.root.parent != second.root.parent


def test_data_manager_lists_and_archives_scoped_experiment_artifacts(tmp_path):
    data = tmp_path / "data"
    runs = data / "runs"
    run = runs / "study-a/evaluate/model-a/task-a/ablation-1/run-1"
    run.mkdir(parents=True)
    (run / "manifest.completed.json").write_text("{}", encoding="utf-8")
    export_node = data / "exports/study-a/evaluate/model-a/task-a/ablation-1"
    interactive = export_node / "interactive/run-1/profile/build-1"
    interactive.mkdir(parents=True)
    (interactive / "manifest.json").write_text("{}", encoding="utf-8")
    isolated = export_node / "isolated/run-1"
    isolated.mkdir(parents=True)
    (isolated / "metadata.json").write_text(json.dumps({
        "schema_version": "ovlab.export-metadata/v2", "export_type": "isolated",
    }), encoding="utf-8")
    manager = DataManager(data, runs)
    items = manager.list()["items"]
    assert {(item["kind"], item["id"]) for item in items} == {
        ("run", "study-a/evaluate/model-a/task-a/ablation-1/run-1"),
        ("report", "study-a/evaluate/model-a/task-a/ablation-1/interactive/run-1"),
        ("export", "isolated:study-a/evaluate/model-a/task-a/ablation-1/isolated/run-1"),
    }
    assert manager.select(run_id="run-1")[0].path == run
    assert manager.select(report_id="study-a/evaluate/model-a/task-a/ablation-1/interactive/run-1")[0].path == interactive.parent.parent
    archived = manager.archive(all_data=True)
    assert archived["target_count"] == 3
    assert (data / "archive/runs/study-a/evaluate/model-a/task-a/ablation-1/run-1/manifest.completed.json").is_file()
    assert (data / "archive/exports/study-a/evaluate/model-a/task-a/ablation-1/interactive/run-1/profile/build-1/manifest.json").is_file()
    assert (data / "archive/exports/study-a/evaluate/model-a/task-a/ablation-1/isolated/run-1/metadata.json").is_file()


def test_data_manager_rejects_ambiguous_bare_scoped_run_id(tmp_path):
    data = tmp_path / "data"
    for experiment in ("study-a", "study-b"):
        run = data / "runs" / experiment / "evaluate" / "run-1"
        run.mkdir(parents=True)
        (run / "manifest.completed.json").write_text("{}", encoding="utf-8")
    manager = DataManager(data, data / "runs")
    with pytest.raises(DataSafetyError, match="ambiguous"):
        manager.select(run_id="run-1")
    assert manager.select(run_id="study-b/evaluate/run-1")[0].path == data / "runs/study-b/evaluate/run-1"


def test_data_manager_discovers_train_grouped_export_at_parent_node(tmp_path):
    data = tmp_path / "data"
    parent = data / "exports/study-a/train/model-a"
    grouped = parent / "grouped/ablations"
    grouped.mkdir(parents=True)
    (grouped / "metadata.json").write_text(json.dumps({
        "schema_version": "ovlab.training-export/v1", "export_type": "grouped",
    }), encoding="utf-8")
    dashboard = parent / "ablation-1/interactive/train-1/system-performance/build-1"
    dashboard.mkdir(parents=True)
    (dashboard / "manifest.json").write_text("{}", encoding="utf-8")
    manager = DataManager(data, data / "runs")
    items = manager.list(kind="all")["items"]
    assert {(item["kind"], item["id"]) for item in items} == {
        ("export", "grouped:study-a/train/model-a/grouped/ablations"),
        ("report", "study-a/train/model-a/ablation-1/interactive/train-1"),
    }
    assert manager.select(export_id="grouped:study-a/train/model-a/grouped/ablations")[0].path == grouped
    assert manager.select(export_id="grouped:study-a/train/model-a/grouped/ablations")[0].state == "generated"


def test_training_deployment_prepares_each_scoped_parent_for_reporting_worker(tmp_path):
    base = tmp_path / "exports"
    base.mkdir()
    sibling = base / "another-study"
    sibling.mkdir()
    sibling.chmod(0o750)
    scope = ExperimentScope("study-a", ("model-a", "task-a", "ablation-1"))
    node = TrainingDeployment._prepare_scoped_hierarchy(base, scope, "train")
    assert node == scope.exports_root(base, "train")
    for directory in (base / "study-a", base / "study-a/train", base / "study-a/train/model-a",
                      base / "study-a/train/model-a/task-a", node):
        assert directory.stat().st_mode & 0o7777 == 0o2770
    assert sibling.stat().st_mode & 0o7777 == 0o750

