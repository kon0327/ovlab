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
