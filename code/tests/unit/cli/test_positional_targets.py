"""Positionals and named aliases address the same CLI operations."""

from pathlib import Path
import json
import os
import subprocess

import pytest

import ovlab_benchctl.cli as cli


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("arguments,method,expected", [
    (["report", "publish", "run-1"], "report_publish", "run-1"),
    (["report", "verify", "run-1"], "report_verify", "run-1"),
    (["export", "isolated", "run-1"], "export_isolated", "run-1"),
    (["export", "grouped", "group-1", "--runs", "run-1", "run-2"], "export_grouped", "group-1"),
    (["export", "grouped", "group-1", "run-1", "run-2"], "export_grouped", "group-1"),
    (["export", "grouped", "--name", "group-1", "run-1", "run-2"], "export_grouped", "group-1"),
    (["export", "generate", "export.yaml"], "export_generate", "export.yaml"),
    (["export", "verify", "group-1"], "export_verify", "grouped"),
    (["data", "list", "runs"], "data_list", None),
    (["dataset", "resolve", "libero_10", "--benchmark", "libero"], "dataset_resolve", None),
    (["dataset", "fetch", "libero_10", "--source", "libero"], "dataset_fetch", None),
    (["dataset", "import", "/input", "--name", "my-data", "--version", "1"], "dataset_import", None),
    (["dataset", "prepare", "data-1", "--format", "openvla-rlds"], "dataset_prepare", "data-1"),
    (["dataset", "inspect", "data-1"], "dataset_inspect", "data-1"),
    (["dataset", "verify", "data-1"], "dataset_verify", "data-1"),
    (["train", "validate", "training.yaml"], "train_validate", "training.yaml"),
    (["train", "plan", "training.yaml"], "train_plan", "training.yaml"),
    (["train", "ablate", "validate", "suite.yaml"], "train_ablate", "suite.yaml"),
    (["train", "report", "train-1"], "train_report", "train-1"),
    (["checkpoint", "inspect", "checkpoint-1"], "checkpoint_inspect", "checkpoint-1"),
    (["checkpoint", "verify", "checkpoint-1"], "checkpoint_verify", "checkpoint-1"),
])
def test_positional_target_reaches_the_correct_operation(monkeypatch, arguments, method, expected):
    calls = []

    class Application:
        def __getattr__(self, name):
            def invoke(*args, **kwargs):
                calls.append((name, args, kwargs))
                return {"ok": True}
            return invoke

    monkeypatch.setattr(cli, "_application", lambda: Application())
    cli._dispatch(cli._parser().parse_args(arguments))
    assert calls[0][0] == method
    if expected is not None:
        assert calls[0][1][0] == expected
    if method == "dataset_fetch":
        assert calls[0][2]["name"] == "libero_10"
    if method == "dataset_import":
        assert calls[0][2]["path"] == "/input"
    if method == "data_list":
        assert calls[0][2]["kind"] == "runs"
    if method == "export_grouped":
        assert calls[0][2]["run_ids"] == ["run-1", "run-2"]


def test_grouped_export_rejects_ambiguous_or_missing_run_selection():
    with pytest.raises(cli.CliUsageError, match="selector flag, not both"):
        cli._dispatch(cli._parser().parse_args(["export", "grouped", "group", "one", "--runs", "two"]))
    with pytest.raises(cli.CliUsageError, match="requires RUNs"):
        cli._dispatch(cli._parser().parse_args(["export", "grouped", "group"]))


@pytest.mark.parametrize("arguments", [
    ["experiment", "plan", "one.json", "--spec", "two.json"],
    ["report", "verify", "one", "--run", "two"],
    ["export", "isolated", "one", "--run", "two"],
    ["export", "verify", "one", "--name", "two"],
    ["dataset", "inspect", "one", "--dataset", "two"],
    ["train", "plan", "one", "--profile", "two"],
    ["checkpoint", "verify", "one", "--checkpoint", "two"],
    ["doctor", "one", "--experiment", "two"],
    ["data", "list", "runs", "--kind", "reports"],
])
def test_duplicate_targets_are_rejected(arguments):
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args(arguments)
    assert exc.value.code == 2


def test_report_generate_is_not_a_cli_command():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["report", "generate", "run-1"])


@pytest.mark.parametrize("target,field,value", [
    ("run:train:run-1", "run_id", "train:run-1"),
    ("report:training:run-1", "report_id", "training:run-1"),
    ("export:isolated:run-1", "export_id", "isolated:run-1"),
])
def test_data_targets_require_a_type_before_preview(monkeypatch, target, field, value):
    calls = []
    class Application:
        def data_preview(self, action, **kwargs):
            calls.append((action, kwargs))
            return {"target_count": 1}
    monkeypatch.setattr(cli, "_application", lambda: Application())
    cli._dispatch(cli._parser().parse_args(["data", "delete", target, "--dry-run"]))
    assert calls[0][1][field] == value
    with pytest.raises(cli.CliUsageError, match="TARGET must"):
        cli._dispatch(cli._parser().parse_args(["data", "delete", "run-1", "--dry-run"]))
    with pytest.raises(cli.CliUsageError, match="not both"):
        cli._dispatch(cli._parser().parse_args(["data", "delete", "run:one", "--run", "two", "--dry-run"]))


def test_experiment_positional_plan_and_named_alias_are_equivalent():
    spec = "experiments/workflows/libero-model-tree.json"
    def invoke(*arguments):
        result = subprocess.run([str(ROOT / "ovlab"), "experiment", "plan", *arguments, "--json"],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)["result"]
    assert invoke(spec) == invoke("--spec", spec)


@pytest.mark.parametrize("action,options,expected", [
    ("plan", [], {"mode": "parallel", "max_workers": 2}),
    ("plan", ["--sequential"], {"mode": "sequential", "max_workers": 1}),
    ("run", ["--dry-run", "--max-workers", "4"], {"mode": "parallel", "max_workers": 4}),
    ("run", ["--dry-run", "--sequential"], {"mode": "sequential", "max_workers": 1}),
])
def test_experiment_cli_selects_execution_mode(action, options, expected):
    result = subprocess.run(
        [str(ROOT / "ovlab"), "experiment", action,
         "experiments/workflows/libero-model-tree.yaml", *options, "--json"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["result"]["execution"] == expected


def test_experiment_cli_rejects_conflicting_execution_options():
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args([
            "experiment", "run", "workflow.yaml", "--sequential", "--max-workers", "2",
        ])
    assert exc.value.code == 2


def test_dataset_positional_import_is_mounted_and_rewritten(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker.txt"
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\nprintf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n')
    docker.chmod(0o755)
    source = tmp_path / "input"
    source.mkdir()
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data"), "OVLAB_FAKE_DOCKER_LOG": str(log)}
    result = subprocess.run([str(ROOT / "ovlab"), "dataset", "import", str(source),
                             "--name", "example", "--version", "1"], cwd=ROOT,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = log.read_text().splitlines()
    assert any(f"source={source},target=/imports/source,readonly" in arg for arg in args)
    assert args[-7:] == ["dataset", "import", "/imports/source", "--name", "example", "--version", "1"]


def test_dataset_positional_fetch_creates_the_correct_publication_directory(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker.txt"
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\nprintf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data"), "OVLAB_FAKE_DOCKER_LOG": str(log)}
    result = subprocess.run([str(ROOT / "ovlab"), "dataset", "fetch", "libero_10", "--source", "libero"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data/datasets/libero/libero_10").is_dir()
    assert log.read_text().splitlines()[-5:] == ["dataset", "fetch", "libero_10", "--source", "libero"]


def test_reporting_host_run_path_is_translated_for_docker(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker.txt"
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\nprintf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n')
    docker.chmod(0o755)
    runs = tmp_path / "runs"
    exports = tmp_path / "exports"
    runs.mkdir()
    exports.mkdir()
    source = runs / "study/evaluate/model/run-1"
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "OVLAB_RUNS_ROOT": str(runs), "OVLAB_EXPORTS_ROOT": str(exports),
           "OVLAB_FAKE_DOCKER_LOG": str(log)}
    result = subprocess.run([str(ROOT / "ovlab"), "report", "verify", str(source)],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines()[-3:] == ["report", "verify", "/var/lib/ovlab/runs/study/evaluate/model/run-1"]


def test_report_publish_routes_machine_report_output(monkeypatch):
    calls = []

    class Application:
        def report_publish(self, run, profile, **options):
            calls.append((run, profile, options))
            return {"status": "completed"}

    monkeypatch.setattr(cli, "_application", lambda: Application())
    cli._dispatch(cli._parser().parse_args(["report", "publish", "run-1", "--output", "out.json"]))
    assert calls == [("run-1", "libero-task-default", {"report_enabled": True, "output": "out.json"})]
    with pytest.raises(cli.CliUsageError, match="cannot be combined"):
        cli._dispatch(cli._parser().parse_args(["report", "publish", "run-1", "--output", "out.json", "--task", "task-1"]))
