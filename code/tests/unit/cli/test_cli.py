"""Dependency-light public CLI contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import ovlab_benchctl.cli as cli_module

from ovlab_benchctl.application import OvlabApplication
from ovlab_benchctl.cli import (
    ExitCode, _SUMMARY_FIELDS, _classify, _compact, _configured_artifact_umask,
    _dispatch, _parser,
)
from ovlab_benchctl.datasets import DatasetRequest, DatasetStore
from ovlab_benchctl.run_references import (
    RunReferenceAmbiguousError, RunReferenceUnavailableError,
)
from ovlab_runner import ReportingRendererError


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
    assert all(command in help_result.stdout for command in (
        "config", "policy", "service", "connect", "run", "metrics",
        "dataset", "train", "checkpoint",
    ))
    version = _run("--version")
    assert version.returncode == 0
    assert version.stdout.startswith("ovlab 0.2.0 (revision ")


def test_dataset_version_does_not_trigger_global_cli_version():
    fetched = _parser().parse_args([
        "dataset", "fetch", "--source", "libero", "--name", "libero_10",
    ])
    imported = _parser().parse_args([
        "dataset", "import", "--name", "custom", "--version", "1",
        "--path", "/imports/source",
    ])
    assert fetched.show_version is False and fetched.version == "1"
    assert imported.show_version is False and imported.version == "1"


def test_run_execution_scopes_sigint_and_sigterm_handlers(monkeypatch):
    events = []

    class Application:
        def run(self, config, *, output_root=None):
            events.append(("run", config, output_root))
            return {"status": "completed"}

    monkeypatch.setattr(cli_module, "_application", lambda: Application())
    monkeypatch.setattr(
        cli_module, "_install_interrupt_handlers",
        lambda: events.append(("install",)) or {"previous": "handlers"},
    )
    monkeypatch.setattr(
        cli_module, "_restore_handlers",
        lambda previous: events.append(("restore", previous)),
    )

    result, json_mode, render = _dispatch(
        _parser().parse_args(["run", "configs/experiments/mock-e2e-smoke.yaml"])
    )

    assert result == {"status": "completed"}
    assert json_mode is False and render is None
    assert events == [
        ("install",),
        ("run", "configs/experiments/mock-e2e-smoke.yaml", None),
        ("restore", {"previous": "handlers"}),
    ]


@pytest.mark.parametrize("arguments", [
    ("config", "validate", "config.yaml"),
    ("policy", "list"),
    ("policy", "describe", "config.yaml"),
    ("service", "serve", "config.yaml"),
    ("service", "health", "--socket", "/tmp/policy.sock"),
    ("connect", "config.yaml"),
    ("deploy", "run", "config.yaml"),
    ("run", "config.yaml"),
    ("metrics", "recompute", "/runs/example"),
    ("report", "profiles"),
    ("report", "generate", "--run", "run-1"),
    ("report", "publish", "--run", "run-1"),
    ("report", "verify", "--run", "run-1"),
    ("export", "isolated", "--run", "run-1"),
    ("export", "grouped", "--name", "group", "--all-runs"),
    ("export", "generate", "--spec", "export.yaml"),
    ("export", "verify", "--name", "group"),
    ("data", "list"),
    ("data", "archive", "--run", "run-1"),
    ("data", "delete", "--report", "run-1"),
    ("data", "archive", "--export", "isolated:run-1"),
    ("dataset", "providers"),
    ("dataset", "resolve", "--benchmark", "libero", "--suite", "libero_10"),
    ("dataset", "fetch", "--source", "libero", "--name", "libero_10"),
    ("dataset", "import", "--name", "custom", "--version", "1", "--path", "/data"),
    ("dataset", "prepare", "--dataset", "dataset-1", "--format", "openvla-rlds"),
    ("dataset", "list"),
    ("dataset", "inspect", "--dataset", "dataset-1"),
    ("dataset", "verify", "--dataset", "dataset-1"),
    ("train", "profiles"),
    ("train", "validate", "--profile", "training.yaml"),
    ("train", "plan", "--profile", "training.yaml"),
    ("train", "run", "--profile", "training.yaml"),
    ("train", "status", "--run", "training-1"),
    ("train", "inspect", "--run", "training-1"),
    ("train", "verify", "--run", "training-1"),
    ("train", "report", "--run", "training-1"),
    ("checkpoint", "list"),
    ("checkpoint", "inspect", "--checkpoint", "checkpoint-1"),
    ("checkpoint", "verify", "--checkpoint", "checkpoint-1"),
])
def test_structured_commands_share_detail_and_json_output_controls(arguments):
    detailed = _parser().parse_args([*arguments, "--detail"])
    machine = _parser().parse_args([*arguments, "--json"])
    assert detailed.detail is True and detailed.json is False
    assert machine.detail is False and machine.json is True


def test_detail_and_json_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parser().parse_args(["policy", "list", "--detail", "--json"])


@pytest.mark.parametrize("command", sorted(_SUMMARY_FIELDS))
def test_every_structured_action_has_a_non_json_compact_summary(command):
    fields = _SUMMARY_FIELDS[command]
    document = {}
    cursor = document
    path = fields[0][1][0].split(".")
    for part in path[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[path[-1]] = "example"

    rendered = _compact(command, document)

    assert rendered
    assert not rendered.lstrip().startswith("{")


def test_dataset_list_defaults_to_compact_rows_and_detail_preserves_document(tmp_path):
    model_data = tmp_path / "model-data"
    source = tmp_path / "source"
    source.mkdir()
    (source / "samples.bin").write_bytes(b"samples")
    registered = DatasetStore(model_data).import_local(DatasetRequest(
        source="local", name="fixture", version="1.0.0", local_path=source,
    ))
    environment = {
        "OVLAB_DATASET_RUNTIME": "host",
        "OVLAB_MODEL_DATA_ROOT": str(model_data),
    }

    compact = _run("dataset", "list", environment=environment)
    assert compact.returncode == 0 and compact.stderr == ""
    assert compact.stdout == f"fixture 1.0.0 {registered['host_path']}\n"

    detailed = _run("dataset", "list", "--detail", environment=environment)
    assert detailed.returncode == 0 and detailed.stderr == ""
    detail_document = json.loads(detailed.stdout)
    assert detail_document["schema_version"] == "ovlab.dataset-list/v1"
    assert detail_document["datasets"][0]["dataset_id"] == registered["dataset_id"]

    machine = _run("dataset", "list", "--json", environment=environment)
    assert machine.returncode == 0 and machine.stderr == ""
    machine_document = json.loads(machine.stdout)
    assert machine_document["schema_version"] == "ovlab-cli-output/1.0.0"
    assert machine_document["result"] == detail_document


def test_policy_list_uses_compact_rows_and_detail_retains_every_field():
    compact = _run("policy", "list")
    assert compact.returncode == 0 and compact.stderr == ""
    assert "vanilla openvla openvla_vanilla -\n" in compact.stdout
    assert "quic-peft openvla_quic quic-peft QP0\n" in compact.stdout

    detailed = _run("policy", "list", "--detail")
    document = json.loads(detailed.stdout)
    assert detailed.returncode == 0 and detailed.stderr == ""
    assert {row["id"] for row in document} == {
        "vanilla", "lora", "openvla-oft", "quic-peft", "quic-wc",
    }


def test_compact_validation_and_empty_registry_outputs_are_operator_friendly(tmp_path):
    validated = _run(
        "config", "validate", QUIC_PEFT,
        "--mode", "descriptor",
    )
    assert validated.returncode == 0 and validated.stderr == ""
    assert "valid" in validated.stdout and "yes" in validated.stdout
    assert "scientific hash" in validated.stdout

    environment = {
        "OVLAB_DATASET_RUNTIME": "host",
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "model-data"),
    }
    checkpoints = _run("checkpoint", "list", environment=environment)
    assert checkpoints.returncode == 0 and checkpoints.stderr == ""
    assert checkpoints.stdout == "No checkpoints found.\n"


def test_source_launcher_uses_dedicated_reporting_container(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_python = binaries / "python-without-matplotlib"
    fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    log = tmp_path / "docker-arguments.txt"
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    data = tmp_path / "data"
    (data / "runs").mkdir(parents=True)
    environment = {
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "OVLAB_PYTHON": str(fake_python),
        "OVLAB_DATA_ROOT": str(data),
        "OVLAB_FAKE_DOCKER_LOG": str(log),
        "OVLAB_REPORTING_IMAGE": "example/ovlab-reporting:test",
    }
    result = _run("export", "isolated", "--run", "run-id", environment=environment)
    assert result.returncode == 0
    assert "using isolated image example/ovlab-reporting:test" in result.stderr
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "run"
    assert "--network" in arguments and "none" in arguments
    assert "OVLAB_RUNS_ROOT=/var/lib/ovlab/runs" in arguments
    assert "OVLAB_DERIVED_ROOT=/var/lib/ovlab/derived" in arguments
    assert "OVLAB_EXPORTS_ROOT=/var/lib/ovlab/exports" in arguments
    assert f"OVLAB_DERIVED_DISPLAY_ROOT={data / 'derived'}" in arguments
    assert f"OVLAB_EXPORTS_DISPLAY_ROOT={data / 'exports'}" in arguments
    assert any("target=/var/lib/ovlab/runs,readonly" in value for value in arguments)
    assert any("target=/var/lib/ovlab/derived" in value for value in arguments)
    assert any("target=/var/lib/ovlab/exports" in value for value in arguments)
    assert arguments[-4:] == ["export", "isolated", "--run", "run-id"]
    assert (data / "exports").stat().st_mode & 0o7777 == 0o2770
    assert (data / "derived").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/isolated").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/grouped").stat().st_mode & 0o7777 == 0o2770


def test_report_profiles_uses_read_only_registry_container_without_data_directories(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    data = tmp_path / "data"

    result = _run(
        "report", "profiles", "--detail",
        environment={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "OVLAB_DATA_ROOT": str(data),
            "OVLAB_FAKE_DOCKER_LOG": str(log),
            "OVLAB_REPORTING_IMAGE": "example/ovlab-reporting:test",
        },
    )

    assert result.returncode == 0
    assert "read-only registry" in result.stderr
    assert not data.exists()
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert "none" in arguments and "--read-only" in arguments
    assert not any("/var/lib/ovlab" in value for value in arguments)
    assert arguments[-3:] == ["report", "profiles", "--detail"]


@pytest.mark.parametrize(
    ("arguments", "artifact_directory", "target"),
    [
        (("report", "verify", "--run", "run-1"), "derived", "/var/lib/ovlab/derived"),
        (("export", "verify", "--kind", "isolated", "--name", "run-1"), "exports", "/var/lib/ovlab/exports"),
    ],
)
def test_reporting_verification_uses_read_only_artifacts_without_chmod(
    tmp_path, arguments, artifact_directory, target,
):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    data = tmp_path / "data"
    (data / "runs").mkdir(parents=True)
    artifact = data / artifact_directory
    artifact.mkdir()
    artifact.chmod(0o555)

    result = _run(
        *arguments,
        environment={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "OVLAB_DATA_ROOT": str(data),
            "OVLAB_FAKE_DOCKER_LOG": str(log),
            "OVLAB_REPORTING_IMAGE": "example/ovlab-reporting:test",
        },
    )

    assert result.returncode == 0
    assert "read-only" in result.stderr
    assert artifact.stat().st_mode & 0o777 == 0o555
    docker_arguments = log.read_text(encoding="utf-8").splitlines()
    assert any(f"target={target},readonly" in value for value in docker_arguments)


@pytest.mark.parametrize("arguments", [
    ("--help",),
    ("fetch", "--help"),
    ("import", "--help"),
])
def test_dataset_help_is_forwarded_offline_without_mutation(tmp_path, arguments):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_python = binaries / "python3"
    fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    log = tmp_path / "docker-arguments.txt"
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    data = tmp_path / "model-data"
    result = _run("dataset", *arguments, environment={
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "OVLAB_PYTHON": str(fake_python),
        "OVLAB_MODEL_DATA_ROOT": str(data),
        "OVLAB_FAKE_DOCKER_LOG": str(log),
        "OVLAB_DATASET_IMAGE": "example/ovlab-dataset:test",
    })
    assert result.returncode == 0
    logged_arguments = log.read_text(encoding="utf-8").splitlines()
    assert logged_arguments[0] == "run"
    assert "--network" in logged_arguments and "none" in logged_arguments
    assert not any("target=/var/lib/ovlab/model-data/datasets" in value for value in logged_arguments)
    assert logged_arguments[-(len(arguments) + 1):] == ["dataset", *arguments]
    assert not data.exists()


def test_dataset_prepare_mounts_existing_registry_read_only(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    model_data = tmp_path / "model-data"
    datasets = model_data / "datasets"
    datasets.mkdir(parents=True)
    datasets.chmod(0o555)

    result = _run(
        "dataset", "prepare", "--dataset", "dataset-example", "--format", "openvla-rlds",
        environment={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "OVLAB_MODEL_DATA_ROOT": str(model_data),
            "OVLAB_FAKE_DOCKER_LOG": str(log),
            "OVLAB_DATASET_IMAGE": "example/ovlab-dataset:test",
        },
    )

    assert result.returncode == 0
    assert datasets.stat().st_mode & 0o777 == 0o555
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert any("target=/var/lib/ovlab/model-data/datasets,readonly" in value for value in arguments)


def test_dataset_fetch_repairs_only_group_writable_publication_parents(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_docker = binaries / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = image ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OVLAB_FAKE_DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    log = tmp_path / "docker-arguments.txt"
    model_data = tmp_path / "model-data"
    name_root = model_data / "datasets/libero/libero_10"
    name_root.mkdir(parents=True)
    (model_data / "datasets/libero").chmod(0o750)
    name_root.chmod(0o750)

    result = _run(
        "dataset", "fetch", "--source", "libero", "--name", "libero_10",
        environment={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "OVLAB_MODEL_DATA_ROOT": str(model_data),
            "OVLAB_FAKE_DOCKER_LOG": str(log),
            "OVLAB_DATASET_IMAGE": "example/ovlab-dataset:test",
        },
    )

    assert result.returncode == 0
    assert (model_data / "datasets/libero").stat().st_mode & 0o7777 == 0o2770
    assert name_root.stat().st_mode & 0o7777 == 0o2770
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert "bridge" in arguments
    assert any("target=/var/lib/ovlab/model-data/datasets" in value for value in arguments)


def test_dataset_launcher_rejects_unsafe_name_before_filesystem_use(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_docker = binaries / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    model_data = tmp_path / "model-data"

    result = _run(
        "dataset", "fetch", "--source", "libero", "--name", "../escaped",
        environment={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "OVLAB_MODEL_DATA_ROOT": str(model_data),
            "OVLAB_DATASET_IMAGE": "example/ovlab-dataset:test",
        },
    )

    assert result.returncode == 3
    assert "path-safe identifiers" in result.stderr
    assert not (tmp_path / "escaped").exists()


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


def test_reporting_renderer_failure_has_typed_exit_code():
    assert _classify(ReportingRendererError("broken template")) == (
        ExitCode.RUNTIME, "report_renderer_error",
    )


def test_run_reference_failures_have_stable_typed_exit_codes():
    assert _classify(RunReferenceUnavailableError("missing")) == (
        ExitCode.POLICY_UNAVAILABLE, "source_unavailable",
    )


def test_artifact_umask_is_scoped_to_one_cli_operation(tmp_path, monkeypatch):
    previous = os.umask(0o022)
    os.umask(previous)
    monkeypatch.setenv("OVLAB_ARTIFACT_UMASK", "0007")
    with _configured_artifact_umask():
        directory = tmp_path / "artifact"
        directory.mkdir()
        file = directory / "value.json"
        file.write_text("{}", encoding="utf-8")
    assert directory.stat().st_mode & 0o777 == 0o770
    assert file.stat().st_mode & 0o777 == 0o660
    observed = os.umask(previous)
    os.umask(observed)
    assert observed == previous
    assert _classify(RunReferenceAmbiguousError("collision")) == (
        ExitCode.USAGE, "ambiguous_run_reference",
    )


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


def test_container_image_tag_does_not_change_scientific_hash():
    profile = str(REPOSITORY / "configs/local/gate-b-showrack.yaml")
    first = OvlabApplication(REPOSITORY, environment={
        **os.environ,
        "OVLAB_LOCAL_PROFILE": profile,
        "OVLAB_BENCHMARK_IMAGE": "registry.example/ovlab-benchmark:first",
        "OVLAB_IMAGE_REFERENCE": "registry.example/ovlab-benchmark:first",
    }).execution_plan("configs/experiments/mock-e2e-smoke.yaml")
    second = OvlabApplication(REPOSITORY, environment={
        **os.environ,
        "OVLAB_LOCAL_PROFILE": profile,
        "OVLAB_BENCHMARK_IMAGE": "registry.example/ovlab-benchmark:second",
        "OVLAB_IMAGE_REFERENCE": "registry.example/ovlab-benchmark:second",
    }).execution_plan("configs/experiments/mock-e2e-smoke.yaml")
    assert first["scientific_config_hash"] == second["scientific_config_hash"]


def test_gate_i_cli_human_and_json_inspection_stay_dependency_light(tmp_path):
    environment = {
        "OVLAB_DATASET_RUNTIME": "host",
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "model-data"),
    }
    human = _run("dataset", "providers", environment=environment)
    assert human.returncode == 0
    assert "libero 1.0.0" in human.stdout
    assert human.stderr == ""

    resolved = _run(
        "dataset", "resolve", "--benchmark", "libero", "--suite", "libero_10", "--json",
        environment=environment,
    )
    assert resolved.returncode == 0 and resolved.stderr == ""
    payload = json.loads(resolved.stdout)
    assert payload["status"] == "success"
    assert payload["result"]["source_revision"] == "a7c9ae18499b6eea8a32f78a9302327b752b1b5f"

    validated = _run(
        "train", "validate", "--profile", "configs/training/openvla-libero10-lora-smoke.yaml", "--json",
        environment=environment,
    )
    result = json.loads(validated.stdout)["result"]
    assert validated.returncode == 0 and validated.stderr == ""
    assert result["valid"] is True
    assert result["model_initialized"] is False
    assert result["network_used"] is False

    checkpoints = _run("checkpoint", "list", "--json", environment=environment)
    assert checkpoints.returncode == 0 and checkpoints.stderr == ""
    assert json.loads(checkpoints.stdout)["result"]["checkpoints"] == []


def test_dataset_registry_does_not_require_an_experiment_config_tree(tmp_path):
    app = OvlabApplication(tmp_path, environment={"OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data")})
    providers = app.dataset_providers()["providers"]
    assert {provider["id"] for provider in providers} == {"libero", "local", "url"}
    assert app._resolver is None


def test_gate_i_plan_missing_dataset_is_typed_and_never_downloads(tmp_path):
    environment = {
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "model-data"),
        "OVLAB_AVAILABLE_GPU_COUNT": "1",
        "OVLAB_AVAILABLE_VRAM_GIB": "32",
    }
    result = _run(
        "train", "plan", "--profile", "configs/training/openvla-libero10-lora-smoke.yaml", "--json",
        environment=environment,
    )
    assert result.returncode == ExitCode.POLICY_UNAVAILABLE
    error = json.loads(result.stdout)["errors"][0]
    assert error["code"] == "dataset_unavailable"
    assert "--allow-dataset-download" in error["message"]
    assert not (tmp_path / "model-data/datasets/.staging").exists()


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
