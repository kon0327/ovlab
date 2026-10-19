"""Host diagnostics stay read-only and identify incompatible deployment images."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from ovlab_benchctl import cli
from ovlab_benchctl.doctor import Doctor


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = "configs/experiments/libero10-lora-merged-rpc-smoke.yaml"


def _docker(command, **kwargs):
    assert kwargs["timeout"] == 5
    if command[1:3] == ["compose", "version"]:
        return subprocess.CompletedProcess(command, 0, "2.30.0\n", "")
    if command[1] == "info":
        return subprocess.CompletedProcess(command, 0, "27.0.0\n", "")
    contracts = {
        "ovlab-benchmark-libero:local": "resolved-checkpoint-config-bundle-v2",
        "ovlab-policy-openvla:local": "resolved-checkpoint-quantization-config-bundle-v2",
        "ovlab-reporting:local": "canonical-readonly-reporting-v1",
    }
    assert command[1:3] == ["image", "inspect"]
    assert 'io.github.ovlab.deployment.contract' in command[4]
    return subprocess.CompletedProcess(command, 0, contracts[command[-1]] + "\n", "")


def test_host_diagnosis_is_read_only_and_reports_docker_failure(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.0\n")
    (tmp_path / "configs").mkdir()
    compose = tmp_path / "deploy/compose"
    compose.mkdir(parents=True)
    (compose / "compose.yaml").write_text("services: {}\n")
    (compose / ".env").write_text("OVLAB_RUNS_ROOT=/tmp/runs\n")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    report = Doctor(tmp_path, which=lambda name: None).run()

    assert report["status"] == "error"
    assert {item["name"]: item["status"] for item in report["checks"]}["docker"] == "error"
    assert report["counts"]["skipped"] == 2
    assert before == sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))


def test_selected_experiment_checks_config_paths_and_current_image_labels(tmp_path):
    runs = tmp_path / "runs"
    exports = tmp_path / "exports"
    datasets = tmp_path / "datasets"
    for path in (runs, exports, datasets):
        path.mkdir()
    report = Doctor(
        ROOT,
        environment={
            "OVLAB_RUNS_ROOT": str(runs),
            "OVLAB_EXPORTS_ROOT": str(exports),
            "OVLAB_DATASETS_PATH": str(datasets),
        },
        runner=_docker,
        which=lambda name: "/usr/bin/docker",
    ).run(experiment=EXPERIMENT, env_file="deploy/compose/.env.example")

    assert report["status"] == "ok"
    assert {item["name"]: item["status"] for item in report["checks"]}["experiment"] == "ok"
    assert sum(item["name"].startswith("image:") for item in report["checks"]) == 3


def test_doctor_json_preserves_findings_and_failure_exit_code(monkeypatch, capsys):
    from ovlab_benchctl import doctor

    monkeypatch.setattr(doctor.Doctor, "run", lambda self, **kwargs: {
        "status": "error", "counts": {"ok": 0, "warning": 0, "error": 1, "skipped": 0},
        "checks": [{"name": "daemon", "status": "error", "message": "unavailable", "remedy": "Start Docker."}],
    })
    assert cli.main(["doctor", "--json"]) == cli.ExitCode.RUNTIME
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "doctor"
    assert output["status"] == "error"
    assert output["result"]["checks"][0]["remedy"] == "Start Docker."
    assert output["errors"][0]["context"]["check"] == "daemon"
