from __future__ import annotations

from pathlib import Path

import pytest

from ovlab_benchctl.application import OvlabApplication
from ovlab_benchctl.run_references import (
    RunReferenceAmbiguousError,
    RunReferenceUnavailableError,
    resolve_run_reference,
    run_hash,
)


def _run(root: Path, identifier: str) -> Path:
    path = root / identifier
    path.mkdir(parents=True)
    return path


def test_current_run_suffix_is_the_displayed_hash_and_legacy_alias_is_stable():
    current = "libero10-oft_2026-07-30_15-18-22_fd24dff0"
    assert run_hash(current) == "fd24dff0"
    assert run_hash("legacy-run") == run_hash("legacy-run")
    assert len(run_hash("legacy-run")) == 8


def test_scoped_training_run_id_keeps_readable_artifact_and_unique_lookup_key():
    from ovlab_core.experiment_scope import ExperimentScope
    from ovlab_benchctl.training_runs import _run_id
    from ovlab_runner.artifacts.layout import run_key

    scope = ExperimentScope("study", ("model", "ablation"))
    first = _run_id("training-profile", "training-plan-12345678", scope)
    second = _run_id("training-profile", "training-plan-12345678", scope)
    assert first != second
    assert run_key(first) == first
    assert run_key(second) == second
    assert run_hash(first) == first[-8:]


def test_resolver_accepts_exact_id_or_hash_and_exact_id_wins(tmp_path):
    root = tmp_path / "runs"
    identifier = "experiment_2026-07-30_15-18-22_fd24dff0"
    path = _run(root, identifier)
    _run(root, "fd24dff0")

    exact = resolve_run_reference(root, identifier)
    hash_named_exact = resolve_run_reference(root, "fd24dff0")

    assert exact.run_id == identifier and exact.path == path.resolve()
    assert hash_named_exact.run_id == "fd24dff0"


def test_hash_collision_is_rejected_instead_of_selecting_a_run(tmp_path):
    root = tmp_path / "runs"
    _run(root, "first_2026-07-30_15-18-22_deadbeef")
    _run(root, "second_2026-07-30_15-18-23_deadbeef")

    with pytest.raises(RunReferenceAmbiguousError, match="matching IDs"):
        resolve_run_reference(root, "deadbeef")
    with pytest.raises(RunReferenceUnavailableError, match="no run matches"):
        resolve_run_reference(root, "aaaaaaaa")


def test_application_resolves_benchmark_and_training_hash_namespaces(tmp_path):
    data = tmp_path / "data"
    benchmark_id = "benchmark_2026-07-30_15-18-22_1234abcd"
    training_id = "training_2026-07-30_15-18-22_9876fedc"
    _run(data / "runs", benchmark_id)
    _run(data / "training-runs", training_id)
    app = OvlabApplication(tmp_path / "repository", environment={
        "OVLAB_DATA_ROOT": str(data),
        "OVLAB_MODEL_DATA_ROOT": str(data),
    })

    assert app._benchmark_run_id("1234abcd") == benchmark_id
    assert app._benchmark_run_path("1234abcd") == (data / "runs" / benchmark_id).resolve()
    assert app._training_run_id("9876fedc") == training_id


def test_application_routes_nested_experiment_reports_to_interactive_only(tmp_path, monkeypatch):
    import ovlab_runner

    data = tmp_path / "data"
    evaluation = _run(data / "runs" / "exp-1" / "evaluate" / "model-a", "eval-1")
    (evaluation / "manifest.started.json").write_text("{}")
    training = _run(data / "runs" / "exp-1" / "train" / "model-a", "train-1")
    (training / "training-plan.json").write_text("{}")
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    app = OvlabApplication(repository, environment={"OVLAB_DATA_ROOT": str(data)})
    calls = []

    def evaluation_engine(runs_root, interactive_root, profile):
        calls.append(("evaluation-root", Path(interactive_root)))
        return type("Engine", (), {
            "generate": lambda self, run, task_id=None: calls.append(("evaluation-generate", Path(run))) or {"ok": True},
            "verify": lambda self, run, build_id=None: calls.append(("evaluation-verify", Path(run))) or {"ok": True},
        })()

    def training_engine(runs_root, interactive_root):
        calls.append(("training-root", Path(interactive_root)))
        return type("Engine", (), {
            "generate": lambda self, run: calls.append(("training-generate", Path(run))) or {"ok": True},
            "verify": lambda self, run, build_id=None: calls.append(("training-verify", Path(run))) or {"ok": True},
        })()

    monkeypatch.setattr(ovlab_runner, "UnifiedInteractiveReportEngine", evaluation_engine)
    monkeypatch.setattr(ovlab_runner, "TrainingInteractiveReportEngine", training_engine)
    monkeypatch.setattr(app, "_report_profile", lambda _: object())

    assert app.report_generate("eval-1") == {"ok": True}
    assert app.report_verify("eval-1") == {"ok": True}
    assert app.train_report("train-1") == {"ok": True}
    assert app.train_report("train-1", verify=True) == {"ok": True}
    assert calls == [
        ("evaluation-root", data / "exports" / "exp-1" / "evaluate" / "model-a" / "interactive"),
        ("evaluation-generate", evaluation.resolve()),
        ("evaluation-root", data / "exports" / "exp-1" / "evaluate" / "model-a" / "interactive"),
        ("evaluation-verify", evaluation.resolve()),
        ("training-root", data / "exports" / "exp-1" / "train" / "model-a" / "interactive"),
        ("training-generate", training.resolve()),
        ("training-root", data / "exports" / "exp-1" / "train" / "model-a" / "interactive"),
        ("training-verify", training.resolve()),
    ]


def test_application_resolves_hash_before_runner_reporting_and_export_calls(
    tmp_path, monkeypatch,
):
    import ovlab_runner

    data = tmp_path / "data"
    run_id = "benchmark_2026-07-30_15-18-22_1234abcd"
    run_path = _run(data / "runs", run_id)
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    app = OvlabApplication(repository, environment={"OVLAB_DATA_ROOT": str(data)})
    calls = []

    monkeypatch.setattr(
        ovlab_runner, "inspect_run",
        lambda path: calls.append(("inspect", Path(path))) or {"run_id": run_id},
    )
    monkeypatch.setattr(
        ovlab_runner, "verify_run",
        lambda path: calls.append(("verify-run", Path(path))) or {"run_id": run_id},
    )
    monkeypatch.setattr(
        ovlab_runner, "recompute_run_metrics",
        lambda path: calls.append(("metrics", Path(path))) or {"run_id": run_id},
    )
    monkeypatch.setattr(
        ovlab_runner.DerivedReportEngine, "generate",
        lambda self, value, **kwargs: calls.append(("report-generate", value)) or {"run_id": value},
    )
    monkeypatch.setattr(
        ovlab_runner.DerivedReportEngine, "verify",
        lambda self, value, **kwargs: calls.append(("report-verify", value)) or {"run_id": value},
    )
    monkeypatch.setattr(
        ovlab_runner.ExportEngine, "generate_isolated",
        lambda self, value, **kwargs: calls.append(("export-isolated", value)) or {"run_id": value},
    )
    monkeypatch.setattr(
        ovlab_runner.ExportEngine, "generate_grouped",
        lambda self, name, **kwargs: calls.append(("export-grouped", kwargs)) or {"name": name},
    )
    monkeypatch.setattr(
        ovlab_runner.ExportEngine, "verify",
        lambda self, kind, name: calls.append(("export-verify", name)) or {"name": name},
    )

    app.inspect("1234abcd")
    app.verify("1234abcd")
    app.recompute_metrics("1234abcd")
    app.report_generate("1234abcd")
    app.report_verify("1234abcd")
    app.export_isolated("1234abcd")
    app.export_grouped("study", run_ids=("1234abcd",))
    app.export_grouped("study", same_model_as="1234abcd")
    app.export_verify("isolated", "1234abcd")

    assert calls[:3] == [
        ("inspect", run_path.resolve()),
        ("verify-run", run_path.resolve()),
        ("metrics", run_path.resolve()),
    ]
    assert ("report-generate", run_id) in calls
    assert ("report-verify", run_id) in calls
    assert ("export-isolated", run_id) in calls
    grouped = [value for kind, value in calls if kind == "export-grouped"]
    assert grouped[0]["run_ids"] == (run_id,)
    assert grouped[1]["same_model_as"] == run_id
    assert ("export-verify", run_id) in calls


def test_report_publish_calls_shared_report_generator_and_exports(tmp_path, monkeypatch):
    import ovlab_runner

    data = tmp_path / "data"
    evaluation = _run(data / "runs", "eval-1")
    (evaluation / "manifest.started.json").write_text("{}")
    app = OvlabApplication(tmp_path / "repository", environment={"OVLAB_DATA_ROOT": str(data)})
    calls = []
    monkeypatch.setattr(app, "report_generate", lambda run, profile, task_id=None:
                        calls.append(("generate", Path(run), profile, task_id)) or {"output": "/report"})

    class Export:
        def __init__(self, *args, **kwargs):
            pass

        def generate_isolated(self, run):
            calls.append(("export", run))
            return {"output": "/export"}

    monkeypatch.setattr(ovlab_runner, "ExportEngine", Export)
    result = app.report_publish("eval-1", "profile-a", task_id="task-1")
    assert result["report"]["output"] == "/report"
    assert result["isolated_export"]["output"] == "/export"
    assert calls == [("generate", evaluation.resolve(), "profile-a", "task-1"), ("export", "eval-1")]
    calls.clear()
    result = app.report_publish("eval-1", report_enabled=False)
    assert result["report"] is None and calls == [("export", "eval-1")]


def test_report_publish_preserves_machine_report_output(tmp_path, monkeypatch):
    import ovlab_runner

    data = tmp_path / "data"
    evaluation = _run(data / "runs", "eval-1")
    (evaluation / "manifest.started.json").write_text("{}")
    app = OvlabApplication(tmp_path / "repository", environment={"OVLAB_DATA_ROOT": str(data)})
    calls = []
    monkeypatch.setattr(ovlab_runner, "regenerate_report", lambda run, output:
                        calls.append(("generate", Path(run), output)) or {"output": output})

    class Export:
        def __init__(self, *args, **kwargs):
            pass

        def generate_isolated(self, run):
            calls.append(("export", run))
            return {"output": "/export"}

    monkeypatch.setattr(ovlab_runner, "ExportEngine", Export)
    result = app.report_publish("eval-1", output="/report.json")
    assert result["report"]["output"] == "/report.json"
    assert calls == [("generate", evaluation.resolve(), "/report.json"), ("export", "eval-1")]


def test_machine_report_displays_host_export_path(tmp_path, monkeypatch):
    import ovlab_runner

    data = tmp_path / "data"
    evaluation = _run(data / "runs", "eval-1")
    (evaluation / "manifest.started.json").write_text("{}")
    exports = data / "exports"
    app = OvlabApplication(tmp_path / "repository", environment={
        "OVLAB_DATA_ROOT": str(data), "OVLAB_EXPORTS_ROOT": str(exports),
        "OVLAB_EXPORTS_DISPLAY_ROOT": "/host/exports",
    })
    monkeypatch.setattr(ovlab_runner, "regenerate_report",
                        lambda run, output: {"output": str(Path(output).resolve())})
    result = app.report_generate("eval-1", output=exports / "machine")
    assert result["output"] == "/host/exports/machine"
