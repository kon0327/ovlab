"""Scoped comparisons expose coverage and only compare matching tasks."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from helpers.contexts import make_run_context
from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_core.contracts import TaskId
from ovlab_runner import (
    DeterministicClock, ExperimentRunner, ExportEngine, FilesystemRunArtifactStore,
    RunConfigurationSnapshot,
)


def _run(root, name, tasks):
    run_id = f"{name}_2026-11-09_12-00-00_abcdef12"
    snapshot = RunConfigurationSnapshot(
        "schema_version: fixture\n",
        f"schema_version: fixture-resolved\nscientific_config_hash: {'a' * 64}\nexecution_config_hash: {'b' * 64}\n",
        "a" * 64, "b" * 64,
    )
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=5),
        selected_task_ids=tuple(TaskId(f"mock-task-{task}") for task in tasks),
        enabled_metric_ids=("task.success", "task.success_rate", "system.inference_latency"),
        metadata={"experiment_id": name},
    )
    runner = ExperimentRunner(
        plan, TrackingBenchmark(task_count=2), TrackingPolicy(),
        FilesystemRunArtifactStore(root), clock=DeterministicClock(),
        configuration_snapshot=snapshot,
    )
    runner.connect()
    runner.run()
    return run_id


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_scoped_grouped_compares_only_shared_tasks(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    first = _run(runs / "study-1/evaluate/model-a", "first", (0, 1))
    second = _run(runs / "study-1/evaluate/model-b", "second", (1,))
    monkeypatch.setenv("OVLAB_EXPERIMENT_ID", "study-1")
    monkeypatch.delenv("OVLAB_EXPERIMENT_NODE", raising=False)
    engine = ExportEngine(runs, tmp_path / "exports")

    published = engine.generate_grouped("models", run_ids=(first, second))
    target = tmp_path / "exports/study-1/evaluate/grouped/models"
    assert published["integrity"] == "verified"
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["comparison"]["status"] == "comparable"
    assert metadata["comparison"]["compared_task_ids"] == ["mock-task-1"]
    assert len(_rows(target / "tables/task-coverage.csv")) == 4
    compared = _rows(target / "tables/comparison-summary.csv")
    assert {row["run_id"]: row["episodes"] for row in compared} == {first: "1", second: "1"}
    assert {row["run_id"]: row["episodes"] for row in _rows(target / "tables/run-summary.csv")} == {
        first: "2", second: "1",
    }


def test_scoped_grouped_marks_no_common_tasks_without_comparison_figures(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    first = _run(runs / "study-1/evaluate/model-a", "first", (0,))
    second = _run(runs / "study-1/evaluate/model-b", "second", (1,))
    monkeypatch.setenv("OVLAB_EXPERIMENT_ID", "study-1")
    monkeypatch.delenv("OVLAB_EXPERIMENT_NODE", raising=False)
    engine = ExportEngine(runs, tmp_path / "exports")

    engine.generate_grouped("models", run_ids=(first, second))
    target = tmp_path / "exports/study-1/evaluate/grouped/models"
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["comparison"]["status"] == "no_shared_tasks"
    assert not (target / "tables/comparison-summary.csv").exists()
    assert not (target / "figures/success-comparison.png").exists()
    assert _rows(target / "tables/task-coverage.csv")


def test_host_cli_publishes_only_interactive_and_isolated_for_scoped_eval(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    runs = tmp_path / "runs"
    run_id = _run(runs / "study-1/evaluate/model-a/task-0", "eval", (0,))
    exports = tmp_path / "exports"
    environment = {
        **os.environ,
        "OVLAB_PYTHON": sys.executable,
        "OVLAB_REPORTING_RUNTIME": "host",
        "OVLAB_DATA_ROOT": str(tmp_path),
        "OVLAB_RUNS_ROOT": str(runs),
        "OVLAB_EXPORTS_ROOT": str(exports),
    }
    result = subprocess.run(
        [str(repository / "ovlab"), "report", "publish", "--run", run_id, "--json"],
        cwd=repository, env=environment, check=True, capture_output=True, text=True,
    )
    published = json.loads(result.stdout)["result"]
    report = Path(published["report"]["interactive_output"])
    isolated = Path(published["isolated_export"]["output"])
    assert report.is_relative_to(exports / "study-1/evaluate/model-a/task-0/interactive")
    assert (report / "index.html").is_file()
    assert isolated == exports / "study-1/evaluate/model-a/task-0/isolated" / run_id
    assert not list(exports.rglob("derived"))


def test_report_automatically_exports_completed_experiment_as_group(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    runs = tmp_path / "runs"
    first = _run(runs / "study-auto/evaluate/family/model-a", "first", (0, 1))
    second = _run(runs / "study-auto/evaluate/family/model-b", "second", (1,))
    third = _run(runs / "study-auto/evaluate/other", "third", (1,))
    manifest = runs / "study-auto/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "experiment_id": "study-auto", "status": "completed", "nodes": {
            "family/model-a": {"evaluate": {"run_id": first}},
            "family/model-b": {"evaluate": {"run_id": second}},
            "other": {"evaluate": {"run_id": third}},
        },
    }))
    (manifest.parent / "plan.json").write_text(json.dumps({
        "experiment_id": "study-auto", "id": "study", "nodes": [
            {"id": "study", "path": "", "children": ["family", "other"]},
            {"id": "family", "path": "family", "children": ["model-a", "model-b"]},
            {"id": "model-a", "path": "family/model-a", "children": []},
            {"id": "model-b", "path": "family/model-b", "children": []},
            {"id": "other", "path": "other", "children": []},
        ],
    }))
    environment = {
        **os.environ, "OVLAB_PYTHON": sys.executable, "OVLAB_REPORTING_RUNTIME": "host",
        "OVLAB_DATA_ROOT": str(tmp_path), "OVLAB_RUNS_ROOT": str(runs),
        "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
    }
    result = subprocess.run([str(repository / "ovlab"), "report", "study-auto", "--json"],
                            cwd=repository, env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    exports = json.loads(result.stdout)["result"]["exports"]
    groups = {item["node"]: item for item in exports if item["type"] == "grouped"}
    assert set(groups["family"]["runs"]) == {first, second}
    assert set(groups[""]["runs"]) == {first, second, third}
    assert all(item["integrity"] == "verified" for item in exports)
    assert (Path(groups["family"]["output"]) / "tables/run-summary.csv").is_file()
    assert (Path(groups[""]["output"]) / "tables/run-summary.csv").is_file()
