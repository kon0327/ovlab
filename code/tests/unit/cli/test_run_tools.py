"""CLI-facing immutable run inspection and offline metric services."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_benchctl.application import _readable_run_id
from ovlab_runner import (
    DeterministicClock, ExperimentRunner, FilesystemRunArtifactStore,
    RunConfigurationSnapshot, RunIntegrityError, inspect_run, recompute_run_metrics,
    regenerate_report, verify_run,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
REPOSITORY = Path(__file__).resolve().parents[4]


def test_readable_run_id_uses_experiment_local_datetime_and_short_hash():
    created = int(
        datetime(2026, 7, 29, 14, 5, 9, tzinfo=timezone.utc).timestamp()
    ) * 1_000_000_000
    first = _readable_run_id(
        "libero10/openvla oft", created, "nonce-a", timezone=timezone.utc
    )
    repeated = _readable_run_id(
        "libero10/openvla oft", created, "nonce-a", timezone=timezone.utc
    )
    different = _readable_run_id(
        "libero10/openvla oft", created, "nonce-b", timezone=timezone.utc
    )
    assert first == repeated
    assert first.startswith("libero10-openvla-oft_2026-07-29_14-05-09_")
    assert len(first.rsplit("_", 1)[1]) == 8
    assert different.rsplit("_", 1)[1] != first.rsplit("_", 1)[1]


@pytest.fixture
def completed_run(tmp_path):
    store = FilesystemRunArtifactStore(tmp_path)
    runner = ExperimentRunner(
        runner_plan(), TrackingBenchmark(maximum_steps=3), TrackingPolicy(), store,
        clock=DeterministicClock(),
        configuration_snapshot=RunConfigurationSnapshot(
            'schema_version: "0.1.0"\nkind: scientific_experiment\n',
            f'scientific_config_hash: {HASH_A}\nexecution_config_hash: {HASH_B}\n',
            HASH_A, HASH_B,
        ),
    )
    runner.connect(); runner.run()
    return store._run_path(runner.plan.run_context.run_id)


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_run_inspect_and_verify_are_read_only(completed_run):
    before = _tree_hash(completed_run)
    summary = inspect_run(completed_run)
    verified = verify_run(completed_run)
    assert summary["status"] == "completed"
    assert summary["rollout_count"] == 1
    assert summary["trace_schema_version"] == "1.0.0"
    assert verified["integrity"] == "verified"
    assert (completed_run / "reports/report.json").is_file()
    assert (completed_run / "reports/report.txt").is_file()
    assert (completed_run / "integrity.json").is_file()
    assert _tree_hash(completed_run) == before


def test_verify_detects_checksum_mutation(completed_run):
    array = next(completed_run.glob("tasks/*/episodes/*/arrays/*.npy"))
    array.write_bytes(array.read_bytes() + b"tampered")
    with pytest.raises(RunIntegrityError, match="checksum mismatch"):
        verify_run(completed_run)


def test_report_regeneration_is_deterministic_separate_and_read_only(completed_run, tmp_path):
    before = _tree_hash(completed_run)
    output = tmp_path / "derived-report"
    result = regenerate_report(completed_run, output)
    assert result["canonical_semantic_match"] is True
    assert result["source_modified"] is False
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == json.loads(
        (completed_run / "reports/report.json").read_text(encoding="utf-8")
    )
    assert (output / "report.txt").read_bytes() == (completed_run / "reports/report.txt").read_bytes()
    assert _tree_hash(completed_run) == before


def test_integrity_inventory_detects_report_tampering(completed_run):
    report = completed_run / "reports/report.txt"
    report.write_text(report.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="integrity checksum mismatch"):
        verify_run(completed_run)


def test_offline_recomputation_preserves_trace_and_complete_results(completed_run):
    before = _tree_hash(completed_run)
    result = recompute_run_metrics(completed_run)
    assert result["metric_api"] == "ovlab-metrics/offline@0.1.0"
    assert result["all_results_agree"] is True
    assert result["original_trace_modified"] is False
    assert result["episode_count"] == 1
    comparison = result["comparisons"][0]
    assert comparison["recorded"] == comparison["recomputed"]
    assert _tree_hash(completed_run) == before


@pytest.mark.parametrize(("arguments", "expected"), [
    (("run", "inspect"), "completed"),
    (("run", "verify"), "verified"),
    (("metrics", "recompute"), True),
])
def test_run_tools_json_subprocess_contract(completed_run, arguments, expected):
    completed = subprocess.run(
        [str(REPOSITORY / "ovlab"), *arguments, str(completed_run), "--json"],
        cwd=REPOSITORY, env=os.environ.copy(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0 and completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    if arguments == ("run", "inspect"):
        assert payload["result"]["status"] == expected
    elif arguments == ("run", "verify"):
        assert payload["result"]["integrity"] == expected
    else:
        assert payload["result"]["all_results_agree"] is expected
