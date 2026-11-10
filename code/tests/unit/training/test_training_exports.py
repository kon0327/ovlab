import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovlab_core.experiment_scope import ExperimentScope
from ovlab_runner.errors import ArtifactError
from ovlab_runner.training_exports import TrainingExportEngine


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _run(root: Path, name: str, dataset_id: str = "dataset-1") -> Path:
    path = root / name
    path.mkdir(parents=True)
    _json(path / "training-plan.json", {
        "scientific": {"profile": {"id": name, "dataset": {"ref": "libero/libero_10", "tasks": ["task-a"]}},
                       "dataset": {"dataset_id": dataset_id}, "training": {"objective": "masked_l1"}},
    })
    _json(path / "result.json", {"schema_version": "ovlab.training-result/v1", "run_id": name,
                                 "status": "completed", "peak_vram_bytes": 2048,
                                 "performance": {"parameter_counts": {"total": 1000, "trainable": 100}}})
    (path / "metrics.jsonl").write_text("".join(json.dumps({
        "schema_version": "ovlab.training-metric/v1", "global_step": step,
        "training_loss": 1 / step, "gpu_memory_allocated_bytes": 1024 * step,
    }) + "\n" for step in (1, 2)), encoding="utf-8")
    files = [{"path": file.name, "size": file.stat().st_size,
              "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in path.iterdir()]
    _json(path / "manifest.json", {"schema_version": "ovlab.training-run/v1", "status": "completed",
                                   "run_id": name, "files": files})
    return path


def test_training_isolated_and_grouped_exports_are_scoped_and_verified(tmp_path):
    scope = ExperimentScope("study-a", ("model-a", "ablation"))
    runs = tmp_path / "runs"
    sources = scope.runs_root(runs, "train")
    _run(sources, "run-1")
    _run(sources, "run-2")
    engine = TrainingExportEngine(runs, tmp_path / "exports", scope=scope)
    isolated = engine.generate_isolated("run-1")
    assert isolated["integrity"] == "verified"
    isolated_root = Path(isolated["output"])
    assert isolated_root == scope.exports_root(tmp_path / "exports", "train") / "isolated/run-1"
    assert (isolated_root / "metrics.csv").is_file()
    assert (isolated_root / "figures/training-loss.svg").is_file()
    html = (isolated_root / "index.html").read_text(encoding="utf-8")
    assert 'figures/training-loss.svg' in html
    assert 'metrics.csv' in html
    assert 'metadata.json' in html
    metadata = json.loads((isolated_root / "metadata.json").read_text(encoding="utf-8"))
    assert any(item["path"] == "index.html" for item in metadata["files"])
    assert isolated_root.parent.stat().st_mode & 0o7777 == 0o2770
    assert isolated_root.stat().st_mode & 0o7777 == 0o2770
    assert (isolated_root / "metrics.csv").stat().st_mode & 0o777 == 0o640
    assert (isolated_root / "metadata.json").stat().st_mode & 0o777 == 0o640
    assert engine.generate_isolated("run-1")["reused"] is True
    isolated_root.chmod(0o550)
    engine.generate_isolated("run-2")
    assert isolated_root.stat().st_mode & 0o7777 == 0o550
    isolated_root.chmod(0o2770)
    grouped = engine.generate_grouped("comparison", run_ids=["run-1", "run-2"])
    assert grouped["integrity"] == "verified"
    grouped_root = Path(grouped["output"])
    assert grouped_root.parent.stat().st_mode & 0o7777 == 0o2770
    comparison = json.loads((grouped_root / "comparison.json").read_text())
    assert comparison["status"] == "comparable"
    assert set(comparison["paired_final_loss"]) == {"run-1", "run-2"}
    assert not list((tmp_path / "exports").rglob("derived"))
    (isolated_root / "metrics.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        engine.verify("isolated", "run-1")


def test_training_grouped_marks_mismatched_datasets_noncomparable(tmp_path):
    scope = ExperimentScope("study-a")
    runs = tmp_path / "runs"
    _run(scope.runs_root(runs, "train") / "model-a", "run-a", "dataset-a")
    _run(scope.runs_root(runs, "train") / "model-b", "run-b", "dataset-b")
    engine = TrainingExportEngine(runs, tmp_path / "exports", scope=scope)
    grouped = engine.generate_grouped("models", run_ids=["run-a", "run-b"])
    comparison = json.loads((Path(grouped["output"]) / "comparison.json").read_text())
    assert comparison["status"] == "not_comparable"
    assert comparison["paired_final_loss"] is None
    assert {row["dataset_id"] for row in comparison["dataset_coverage"]} == {"dataset-a", "dataset-b"}


def test_existing_training_export_gains_checksum_verified_browser_without_losing_data(tmp_path):
    scope = ExperimentScope("study-a")
    runs = tmp_path / "runs"
    _run(scope.runs_root(runs, "train"), "old-run")
    engine = TrainingExportEngine(runs, tmp_path / "exports", scope=scope)
    root = Path(engine.generate_isolated("old-run")["output"])
    metadata = json.loads((root / "metadata.json").read_text())
    (root / "index.html").unlink()
    metadata["files"] = [item for item in metadata["files"] if item["path"] != "index.html"]
    _json(root / "metadata.json", metadata)
    before = {item["path"]: (root / item["path"]).read_bytes() for item in metadata["files"]}
    assert engine.verify("isolated", "old-run")["integrity"] == "verified"

    upgraded = engine.generate_isolated("old-run")
    assert upgraded["reused"] and upgraded["index_added"]
    assert engine.verify("isolated", "old-run")["integrity"] == "verified"
    assert all((root / name).read_bytes() == contents for name, contents in before.items())
    assert "figures/training-loss.svg" in (root / "index.html").read_text()
    assert not list(root.parent.glob(".old-run.*"))
    assert "index_added" not in engine.generate_isolated("old-run")


def test_host_cli_train_report_publishes_and_verifies_inside_experiment(tmp_path):
    repository = Path(__file__).resolve().parents[4]
    scope = ExperimentScope("study-a", ("model-a", "ablation"))
    _run(scope.runs_root(tmp_path / "runs", "train"), "run-1")
    environment = {**os.environ, "OVLAB_PYTHON": sys.executable, "OVLAB_REPORTING_RUNTIME": "host",
                   "OVLAB_DATA_ROOT": str(tmp_path), "OVLAB_EXPERIMENT_ID": scope.experiment_id,
                   "OVLAB_EXPERIMENT_NODE": scope.node_path}
    generated = subprocess.run([str(repository / "ovlab"), "train", "report", "--run", "run-1", "--json"],
                               cwd=repository, env=environment, text=True, capture_output=True, check=False)
    assert generated.returncode == 0, generated.stderr
    output = Path(json.loads(generated.stdout)["result"]["output"])
    assert output.is_relative_to(scope.exports_root(tmp_path / "exports", "train") / "interactive")
    verified = subprocess.run([str(repository / "ovlab"), "train", "report", "--run", "run-1", "--verify", "--json"],
                              cwd=repository, env=environment, text=True, capture_output=True, check=False)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["result"]["integrity"] == "verified"
