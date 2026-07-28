"""Dependency-light public CLI contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovlab_benchctl.application import OvlabApplication


REPOSITORY = Path(__file__).resolve().parents[4]
OVLAB = REPOSITORY / "ovlab"
QUIC_PEFT = "configs/policies/openvla-quic/quic-peft-bones.yaml"
QUIC_WC = "configs/policies/openvla-quic/quic-wc-bones.yaml"


def _run(*args, environment=None):
    env = os.environ.copy()
    if environment:
        env.update(environment)
    return subprocess.run(
        [str(OVLAB), *args], cwd=REPOSITORY, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


@pytest.mark.parametrize("arguments", [
    ("--help",),
    ("--version",),
    ("policy", "list", "--json"),
    ("config", "validate", QUIC_PEFT, "--mode", "descriptor", "--json"),
])
def test_lightweight_commands_do_not_import_runtime_packages(arguments):
    invocation = "code=0; _parser().format_help()" if arguments == ("--help",) else f"code=main({list(arguments)!r})"
    expression = (
        "import json,sys; from ovlab_benchctl.cli import main,_parser; "
        f"{invocation}; "
        "blocked=('torch','transformers','peft','libero','mujoco','robosuite','openvla_quic'); "
        "print(json.dumps({'code':code,'loaded':sorted(k for k in sys.modules if k.split('.')[0] in blocked)}))"
    )
    env = os.environ.copy()
    paths = [str(path) for path in REPOSITORY.glob("code/*/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("code/apps/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("code/packages/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("code/policies/*/src")]
    env["PYTHONPATH"] = ":".join(paths)
    completed = subprocess.run(
        [sys.executable, "-c", expression], cwd=REPOSITORY, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert '"loaded": []' in completed.stdout


def test_help_and_version_have_stable_identity():
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "Config" not in help_result.stderr
    assert all(command in help_result.stdout for command in ("config", "policy", "service", "connect", "run", "metrics"))
    version = _run("--version")
    assert version.returncode == 0
    assert version.stdout.startswith("ovlab 0.1.0 (revision ")


def test_policy_list_is_static_distinct_and_has_no_non_quic_qp_metadata():
    result = _run("policy", "list", "--json")
    assert result.returncode == 0 and result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    methods = {item["id"]: item for item in payload["result"]}
    assert set(methods) == {"vanilla", "lora", "openvla-oft", "quic-peft", "quic-wc"}
    assert all("quic_profile" not in methods[name] for name in ("vanilla", "lora", "openvla-oft"))
    assert methods["quic-peft"]["quic_profile"] == methods["quic-wc"]["quic_profile"] == "QP0"


@pytest.mark.parametrize("config", [QUIC_PEFT, QUIC_WC])
def test_quic_descriptor_validation_and_description_are_inspection_only(config):
    validated = _run("config", "validate", config, "--mode", "descriptor", "--json")
    assert validated.returncode == 0 and validated.stderr == ""
    described = _run("policy", "describe", config, "--json")
    result = json.loads(described.stdout)["result"]
    assert result["readiness"]["runtime_ready"] is False
    assert result["profile"] == {
        "active_transformation": False,
        "definition_availability": "not_applicable",
        "definition_hash": None,
        "definition_version": None,
        "id": "QP0",
    }


@pytest.mark.parametrize(
    ("config", "error_type"),
    [
        (QUIC_PEFT, "QuICPEFTIntegrationIncompleteError"),
        (QUIC_WC, "QuICWCImplementationIncompleteError"),
    ],
)
def test_quic_runtime_and_dry_run_fail_with_typed_exit_code(config, error_type):
    result = _run("config", "validate", config, "--mode", "runtime", "--json")
    assert result.returncode == 4 and result.stderr == ""
    error = json.loads(result.stdout)["errors"][0]
    assert error["type"] == error_type
    assert error["context"]["variant"] in {"quic-peft", "quic-wc"}
    dry = _run("run", config, "--dry-run", "--json")
    assert dry.returncode == 4
    assert json.loads(dry.stdout)["errors"][0]["type"] == error_type


def test_json_success_and_failure_are_single_documents_with_stderr_purity():
    success = _run("policy", "list", "--json")
    assert success.stdout.count("\n") == 1 and success.stderr == ""
    assert set(json.loads(success.stdout)) == {"schema_version", "command", "status", "result", "errors"}
    failure = _run("config", "validate", "configs/does-not-exist.yaml", "--json")
    assert failure.returncode == 3 and failure.stderr == ""
    assert failure.stdout.count("\n") == 1
    assert json.loads(failure.stdout)["errors"][0]["code"] == "configuration_error"


def test_human_diagnostics_use_stderr_without_traceback():
    result = _run("config", "validate", "configs/does-not-exist.yaml")
    assert result.returncode == 3
    assert result.stdout == ""
    assert "configuration_error" in result.stderr
    assert "Traceback" not in result.stderr


def test_semantic_usage_error_has_stable_exit_code():
    result = _run("run", "inspect")
    assert result.returncode == 2
    assert "usage_error" in result.stderr


def test_dry_run_is_deterministic_and_has_no_filesystem_side_effects(tmp_path):
    profile = tmp_path / "profile.yaml"
    runs = tmp_path / "runs"
    profile.write_text(
        'schema_version: "0.1.0"\nkind: local_profile\nid: cli-test\n\npaths:\n'
        f'  checkpoint_root: {tmp_path}/checkpoints\n  dataset_root: {tmp_path}/datasets\n  runs_root: {runs}\n\n'
        'devices:\n  primary_gpu: cuda:0\n', encoding="utf-8",
    )
    env = {"OVLAB_LOCAL_PROFILE": str(profile)}
    first = _run("run", "configs/experiments/mock-e2e-smoke.yaml", "--dry-run", "--json", environment=env)
    second = _run("run", "configs/experiments/mock-e2e-smoke.yaml", "--dry-run", "--json", environment=env)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["result"]["side_effects_performed"] is False
    assert not runs.exists()
    real = _run("run", "configs/experiments/mock-e2e-smoke.yaml", "--json", environment=env)
    assert real.returncode == 3
    assert "production CLI run currently requires LIBERO" in json.loads(real.stdout)["errors"][0]["message"]
    assert not runs.exists()


def test_resolved_json_is_deterministic_and_not_wrapped_as_python_repr():
    environment = {"OVLAB_LOCAL_PROFILE": str(REPOSITORY / "configs/local/gate-b-showrack.yaml")}
    arguments = (
        "config", "resolve", "configs/experiments/mock-e2e-smoke.yaml",
        "--mode", "runtime", "--format", "json",
    )
    first = _run(*arguments, environment=environment)
    second = _run(*arguments, environment=environment)
    assert first.returncode == second.returncode == 0 and first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    document = json.loads(first.stdout)
    assert document["kind"] == "resolved_experiment"
    assert document["scientific_config_hash"]


def test_output_root_does_not_change_scientific_hash(tmp_path):
    environment = {"OVLAB_LOCAL_PROFILE": str(REPOSITORY / "configs/local/gate-b-showrack.yaml")}
    app = OvlabApplication(REPOSITORY, environment={**os.environ, **environment})
    first = app.execution_plan("configs/experiments/mock-e2e-smoke.yaml", output_root=tmp_path / "one")
    second = app.execution_plan("configs/experiments/mock-e2e-smoke.yaml", output_root=tmp_path / "two")
    assert first["scientific_config_hash"] == second["scientific_config_hash"]
    assert first["output_root"] != second["output_root"]


def test_cli_module_contains_no_rollout_or_metric_implementation():
    source = (REPOSITORY / "code/apps/benchctl/src/ovlab_benchctl/cli.py").read_text(encoding="utf-8")
    assert "execute_episode" not in source
    assert "MetricEvaluator" not in source
    assert "ExperimentRunner(" not in source


def test_application_delegates_to_runner_without_duplicating_episode_loop():
    source = (REPOSITORY / "code/apps/benchctl/src/ovlab_benchctl/application.py").read_text(encoding="utf-8")
    assert "ExperimentRunner(" in source
    assert "runner.connect()" in source and "runner.run()" in source
    assert "execute_episode" not in source
    assert "while step" not in source
