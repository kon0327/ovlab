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
QUIC_POLICY = "configs/.tests/evaluation/quic.yaml"


def test_unified_run_help_shows_one_target_and_only_modes():
    result = _run("run", "--help")
    assert result.returncode == 0
    assert "usage: ovlab run " in result.stdout
    assert "EXPERIMENT" in result.stdout
    assert "--plan" in result.stdout and "--verify" in result.stdout
    for removed in ("--node", "--renderer", "--profile", "--max-workers", "--quantization", "{plan,run}"):
        assert removed not in result.stdout


@pytest.mark.parametrize("source,phase", [
    ("examples/experiments/plugin-pipeline-smoke.yaml", "train"),
    ("configs/.tests/evaluation/openvla.yaml", "evaluate"),
    ("examples/experiments/plugin-pipeline-smoke.yaml", "train"),
])
def test_unified_plan_resolves_all_source_kinds_without_side_effects(source, phase):
    result = _run("run", source, "--plan", "--json")
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["command"] == "run"
    assert document["result"]["side_effects_performed"] is False
    assert document["result"]["steps"][0]["phase"] == phase


@pytest.mark.parametrize("mode", ["--plan", "--verify"])
def test_unified_config_modes_never_start_a_workload(mode):
    result = _run("run", "examples/experiments/plugin-pipeline-smoke.yaml", mode, "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["side_effects_performed"] is False


def test_registry_checkpoint_list_cli(tmp_path):
    environment = {"OVLAB_GLOBAL_HF_CACHE": str(tmp_path / "cache"),
                   "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data")}
    result = _run("checkpoint", "list", "--source", "registry", "--offline", "--json", environment=environment)
    assert result.returncode == 0
    document = json.loads(result.stdout)["result"]
    assert document["source"] == "registry" and document["network"] == "offline"
    assert any(row["checkpoint_id"] == "openvla-7b" for row in document["checkpoints"])
    assert not (tmp_path / "data").exists()
    assert _run("checkpoint", "list", "--online", environment=environment).returncode != 0


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
    ("plugin", "list", "--json"),
])
def test_lightweight_commands_do_not_import_runtime_packages(arguments):
    invocation = "code=0; _parser().format_help()" if arguments == ("--help",) else f"code=main({list(arguments)!r})"
    expression = (
        "import json,sys; from ovlab_benchctl.cli import main,_parser; "
        f"{invocation}; "
        "blocked=('torch','transformers','peft','libero','mujoco','robosuite','openvla_quic','quic_petf'); "
        "print(json.dumps({'code':code,'loaded':sorted(k for k in sys.modules if k.split('.')[0] in blocked)}))"
    )
    env = os.environ.copy()
    paths = [str(path) for path in REPOSITORY.glob("code/*/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("code/apps/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("code/packages/*/src")]
    paths += [str(path) for path in REPOSITORY.glob("plugins/*/src")]
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
        "dataset", "checkpoint",
    ))
    version = _run("--version")
    assert version.returncode == 0
    assert version.stdout.startswith("ovlab 0.2.0 (revision ")


def test_training_summary_separates_completed_checkpoint_from_weak_validation():
    report = {"run_id": "smoke", "host_path": "/runs/smoke", "result": {
        "status": "completed", "checkpoint_id": "checkpoint-1", "optimizer_steps": 3,
        "adapter_parameter_count": 14880, "changed_adapter_tensor_count": 4,
        "identity_logit_max_absolute_error": 0.0, "reload_logit_max_absolute_error": 0.0,
        "adapter_validation": {"action_token_accuracy": 0.0, "action_tokens": 28,
                               "action_token_cross_entropy": 27.2857},
    }}
    summary = _compact("train run", report)
    assert "[OK] checkpoint finalized; 3 optimizer steps; 4 adapter tensors updated; exact save/reload" in summary
    assert "[CHECK] 0.0% on 28 action tokens" in summary
    assert "rollout not measured" in summary
    assert "adapter_parameter_count" not in summary


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


@pytest.mark.parametrize("command", ("eval", "train"))
def test_run_requires_yaml_target_and_rejects_old_profile_option(command):
    parser = _parser()
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([command, "run"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([command, "run", "example", "--profile", "other.yaml"])
    assert parser.parse_args([command, "run", "example"]).target == "example"


@pytest.mark.parametrize("option", ("--env-file", "--local-profile"))
def test_deploy_run_rejects_removed_host_path_options(option):
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(["eval", "run", "example", option, "host.yaml"])


def test_run_execution_scopes_sigint_and_sigterm_handlers(monkeypatch):
    events = []

    class Workflow:
        def __init__(self, root):
            pass
        def run(self, config):
            events.append(("run", config))
            return {"status": "completed"}

    from ovlab_benchctl import experiment_workflow
    monkeypatch.setattr(experiment_workflow, "ExperimentWorkflow", Workflow)
    monkeypatch.setattr(
        cli_module, "_install_interrupt_handlers",
        lambda: events.append(("install",)) or {"previous": "handlers"},
    )
    monkeypatch.setattr(
        cli_module, "_restore_handlers",
        lambda previous: events.append(("restore", previous)),
    )

    result, json_mode, render = _dispatch(
        _parser().parse_args(["run", "examples/experiments/mock-smoke.yaml"])
    )

    assert result == {"status": "completed"}
    assert json_mode is False and render is None
    assert events == [
        ("install",),
        ("run", "examples/experiments/mock-smoke.yaml"),
        ("restore", {"previous": "handlers"}),
    ]


def test_private_native_worker_resolves_bare_target_and_local_profile(monkeypatch):
    class Application:
        def execution_plan(self, config, *, output_root=None, local_profile=None):
            return {"config": str(config), "output_root": output_root, "local_profile": local_profile}

    monkeypatch.setattr(cli_module, "_application", lambda: Application())
    args = _parser().parse_args(["_native-run", "mock-smoke", "--profile", "local.yaml", "--dry-run"])
    result, _, _ = _dispatch(args)
    assert result["config"] == str(REPOSITORY / "examples/experiments/mock-smoke.yaml")
    assert result["local_profile"] == "local.yaml"


def test_unified_run_rejects_missing_target_duplicate_modes_and_runtime_flags():
    for arguments in (["run"], ["run", "example", "--plan", "--verify"],
                      ["run", "inspect", "run-1"], ["run", "example", "--profile", "local.yaml"],
                      ["run", "train", "example"]):
        with pytest.raises(SystemExit):
            _parser().parse_args(arguments)


@pytest.mark.parametrize("arguments", [
    ("config", "validate", "config.yaml"),
    ("policy", "list"),
    ("policy", "describe", "config.yaml"),
    ("service", "serve", "config.yaml"),
    ("service", "health", "--socket", "/tmp/policy.sock"),
    ("connect", "config.yaml"),
    ("eval", "run", "config.yaml"),
    ("run", "config.yaml"),
    ("metrics", "recompute", "/runs/example"),
    ("report", "profiles"),
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
    ("train", "run", "training.yaml"),
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
    assert "DATASET" in compact.stdout and "VERSION" in compact.stdout and "PATH" in compact.stdout
    assert "fixture" in compact.stdout and "1.0.0" in compact.stdout

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
    assert all(header in compact.stdout for header in ("ID", "FAMILY", "CONFIG TYPE", "RUNTIME", "PROFILES", "SOURCES"))
    assert "vanilla" in compact.stdout and "openvla_vanilla" in compact.stdout
    assert "quic-peft" in compact.stdout and "experimental" in compact.stdout

    detailed = _run("policy", "list", "--detail")
    document = json.loads(detailed.stdout)
    assert detailed.returncode == 0 and detailed.stderr == ""
    assert {row["id"] for row in document} == {
        "vanilla", "lora", "openvla-oft", "quic-peft",
    }


def test_compact_validation_and_empty_registry_outputs_are_operator_friendly(tmp_path):
    validated = _run(
        "config", "validate", "ovlab://policies/mock.yaml",
        "--mode", "descriptor",
    )
    assert validated.returncode == 0 and validated.stderr == ""
    assert "valid" in validated.stdout and "yes" in validated.stdout
    assert "policy" in validated.stdout and "descriptor" in validated.stdout

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
    assert "OVLAB_EXPORTS_ROOT=/var/lib/ovlab/exports" in arguments
    assert f"OVLAB_DERIVED_DISPLAY_ROOT={data / 'exports' / 'evaluate' / 'derived'}" in arguments
    assert f"OVLAB_EXPORTS_DISPLAY_ROOT={data / 'exports'}" in arguments
    assert any("target=/var/lib/ovlab/runs,readonly" in value for value in arguments)
    assert any("target=/var/lib/ovlab/exports" in value for value in arguments)
    assert arguments[-4:] == ["export", "isolated", "--run", "run-id"]
    assert (data / "exports").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/evaluate/derived").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/evaluate/interactive").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/evaluate/isolated").stat().st_mode & 0o7777 == 0o2770
    assert (data / "exports/evaluate/grouped").stat().st_mode & 0o7777 == 0o2770


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
        (("report", "verify", "--run", "run-1"), "exports/derived", "/var/lib/ovlab/exports"),
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
    artifact.mkdir(parents=True)
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


def test_policy_list_reports_current_profiles_sources_and_runtime_status():
    from ovlab_benchctl.quic_experiments import PROFILES
    result = _run("policy", "list", "--json")
    assert result.returncode == 0 and result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    methods = {item["id"]: item for item in payload["result"]}
    assert set(methods) == {"vanilla", "lora", "openvla-oft", "quic-peft"}
    assert methods["quic-peft"]["supported_quic_profiles"] == [*PROFILES, "custom"]
    assert methods["quic-peft"]["config_type"] == "openvla_quic"
    assert methods["quic-peft"]["runtime_status"] == "experimental"
    assert methods["quic-peft"]["source_repositories"] == ["external/openvla-quic", "external/quic-peft-vla"]
    assert all("supported_quic_profiles" not in methods[name] for name in ("vanilla", "lora", "openvla-oft"))
    assert all("quic_profile" not in item for item in methods.values())
    human = _run("policy", "list")
    assert human.returncode == 0
    assert all(value in human.stdout for value in ("QP1", "QP2", "QP3", "QP4", "custom"))
    assert "QP0" not in human.stdout


def test_quic_default_requires_checkpoint_binding_or_explicit_adapter():
    result = _run("config", "validate", "plugin://openvla-quic/policy/default.yaml", "--json")
    assert result.returncode == ExitCode.CONFIGURATION
    assert "adapter" in json.loads(result.stdout)["errors"][0]["message"]


def test_complete_quic_policy_can_be_validated_without_model_initialization():
    result = _run("config", "validate", ".tests/components/quic-policy.yaml", "--json")
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout)["result"]["valid"] is True


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
    result = _run("run", "example", "--plan", "--verify")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


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
    first = _run("_native-run", "examples/experiments/mock-smoke.yaml", "--dry-run", "--json", environment=env)
    second = _run("_native-run", "examples/experiments/mock-smoke.yaml", "--dry-run", "--json", environment=env)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["result"]["side_effects_performed"] is False
    assert not runs.exists()
    real = _run("_native-run", "examples/experiments/mock-smoke.yaml", "--json", environment=env)
    assert real.returncode == 3
    assert "production CLI run currently requires LIBERO" in json.loads(real.stdout)["errors"][0]["message"]
    assert not runs.exists()


def test_run_profile_option_overrides_local_profile_environment(tmp_path):
    def local_profile(name):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            'schema_version: "0.1.0"\nkind: local_profile\n'
            f'id: {name}\n\npaths:\n'
            f'  checkpoint_root: {tmp_path}/{name}/checkpoints\n'
            f'  dataset_root: {tmp_path}/{name}/datasets\n'
            f'  runs_root: {tmp_path}/{name}/runs\n\n'
            'devices:\n  primary_gpu: cuda:0\n',
            encoding="utf-8",
        )
        return path

    configured = local_profile("configured")
    selected = local_profile("selected")
    result = _run(
        "_native-run", "mock-smoke", "--profile", str(selected), "--dry-run", "--json",
        environment={"OVLAB_LOCAL_PROFILE": str(configured)},
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)["result"]
    assert plan["output_root"] == str(tmp_path / "selected/runs/evaluate")
    assert not (tmp_path / "selected/runs").exists()


def test_resolved_json_is_deterministic_and_not_wrapped_as_python_repr():
    environment = {"OVLAB_LOCAL_PROFILE": str(REPOSITORY / "configs/local/gate-b-showrack.yaml")}
    arguments = (
        "config", "resolve", "examples/experiments/mock-smoke.yaml",
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
    first = app.execution_plan("examples/experiments/mock-smoke.yaml", output_root=tmp_path / "one")
    second = app.execution_plan("examples/experiments/mock-smoke.yaml", output_root=tmp_path / "two")
    assert first["scientific_config_hash"] == second["scientific_config_hash"]
    assert first["output_root"] != second["output_root"]


def test_container_image_tag_does_not_change_scientific_hash():
    profile = str(REPOSITORY / "configs/local/gate-b-showrack.yaml")
    first = OvlabApplication(REPOSITORY, environment={
        **os.environ,
        "OVLAB_LOCAL_PROFILE": profile,
        "OVLAB_BENCHMARK_IMAGE": "registry.example/ovlab-benchmark:first",
        "OVLAB_IMAGE_REFERENCE": "registry.example/ovlab-benchmark:first",
    }).execution_plan("examples/experiments/mock-smoke.yaml")
    second = OvlabApplication(REPOSITORY, environment={
        **os.environ,
        "OVLAB_LOCAL_PROFILE": profile,
        "OVLAB_BENCHMARK_IMAGE": "registry.example/ovlab-benchmark:second",
        "OVLAB_IMAGE_REFERENCE": "registry.example/ovlab-benchmark:second",
    }).execution_plan("examples/experiments/mock-smoke.yaml")
    assert first["scientific_config_hash"] == second["scientific_config_hash"]


def test_gate_i_cli_human_and_json_inspection_stay_dependency_light(tmp_path):
    environment = {
        "OVLAB_DATASET_RUNTIME": "host",
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "model-data"),
    }
    human = _run("dataset", "providers", environment=environment)
    assert human.returncode == 0
    assert "ID" in human.stdout and "VERSION" in human.stdout
    assert "libero" in human.stdout and "1.0.0" in human.stdout
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
        "train", "validate", "--profile", "configs/.tests/train/lora.yaml", "--json",
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
        "train", "plan", "--profile", "configs/.tests/train/lora.yaml", "--json",
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


@pytest.mark.parametrize("arguments", [
    ["train", "validate", "--profile", "profile.yaml"],
    ["train", "plan", "--profile", "profile.yaml"],
    ["train", "run", "profile.yaml"],
    ["train", "ablate", "validate", "--suite", "suite.yaml"],
    ["train", "ablate", "plan", "--suite", "suite.yaml"],
    ["train", "ablate", "materialize", "--suite", "suite.yaml", "--output", "/tmp/fit-plan"],
    ["train", "ablate", "run", "--suite", "suite.yaml", "--output", "/tmp/fit-run"],
])
def test_training_normalization_override_is_explicit_across_commands(arguments):
    assert _parser().parse_args(arguments).fit_action_normalization is False
    assert _parser().parse_args([*arguments, "--fit-action-normalization"]).fit_action_normalization is True


def test_list_table_adapts_to_terminal_width_without_losing_values(monkeypatch):
    from ovlab_benchctl.cli import _table

    rows = [{"id": "alpha", "state": "ready"}, {"id": "beta", "state": "waiting"}]
    columns = (("ID", ("id",)), ("STATE", ("state",)))
    monkeypatch.setenv("COLUMNS", "80")
    wide = _table(rows, columns, empty="empty")
    assert wide.splitlines()[0].split() == ["ID", "STATE"]
    assert "alpha" in wide and "waiting" in wide

    monkeypatch.setenv("COLUMNS", "12")
    narrow = _table(rows, columns, empty="empty")
    assert "ID     alpha" in narrow and "STATE  ready" in narrow
    assert "ID     beta" in narrow and "STATE  waiti" in narrow and "       ng" in narrow
    assert all(len(line) <= 12 for line in narrow.splitlines())
    assert _table([], columns, empty="No entries.") == "No entries."


def test_list_table_wraps_long_values_without_truncation(monkeypatch):
    from ovlab_benchctl.cli import _table

    monkeypatch.setenv("COLUMNS", "30")
    value = "abcdefghijklmnopqrstuvwxyz0123456789"
    rendered = _table([{"id": value, "state": "ready"}],
                      (("ID", ("id",)), ("STATE", ("state",))), empty="empty")
    assert "".join(line.strip() for line in rendered.splitlines()[0:2]).endswith(value)
    assert all(len(line) <= 30 for line in rendered.splitlines())


@pytest.mark.parametrize(("command", "result", "headings"), [
    ("report profiles", {"profiles": [{"id": "default", "source": "built-in", "template": "base"}]}, ("ID", "SOURCE", "TEMPLATE")),
    ("dataset providers", {"providers": [{"id": "libero", "version": "1"}]}, ("ID", "VERSION")),
    ("dataset list", {"datasets": [{"logical_name": "data", "dataset_version": "1", "path": "/data"}]}, ("DATASET", "VERSION", "PATH")),
    ("train profiles", {"profiles": [{"id": "train", "valid": True, "path": "/train"}]}, ("PROFILE", "VALID", "PATH")),
    ("checkpoint list", {"checkpoints": [{"checkpoint_id": "model", "kind": "adapter", "merge_status": "merged", "host_path": "/model"}]}, ("CHECKPOINT", "KIND", "MERGE STATUS", "PATH")),
    ("checkpoint list", {"source": "registry", "checkpoints": [{"checkpoint_id": "model", "repo_id": "owner/model", "revision": "123", "local_status": "present", "remote_status": "not_checked"}]}, ("CHECKPOINT", "REPOSITORY", "REVISION", "LOCAL", "REMOTE")),
    ("data list", {"items": [{"kind": "run", "id": "run-1", "run_hash": "abc", "state": "completed", "run_type": "train", "path": "/runs"}]}, ("KIND", "ID", "HASH", "STATE", "TYPE", "PATH")),
])
def test_list_commands_include_column_labels(monkeypatch, command, result, headings):
    monkeypatch.setenv("COLUMNS", "240")
    rendered = _compact(command, result)
    assert all(heading in rendered.splitlines()[0] for heading in headings)
    assert len(rendered.splitlines()) >= 3


def test_dataset_container_receives_terminal_width(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    docker = binaries / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n', encoding="utf-8"
    )
    docker.chmod(0o755)
    result = _run("dataset", "list", "--json", environment={
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "OVLAB_FAKE_DOCKER_LOG": str(log),
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data"),
        "COLUMNS": "73",
    })
    assert result.returncode == 0
    arguments = log.read_text().splitlines()
    assert "COLUMNS=73" in arguments
    assert arguments[-3:] == ["dataset", "list", "--json"]


def test_dataset_list_renders_image_json_with_host_table(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    docker = binaries / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n'
        '''printf '%s\\n' '{"status":"success","result":{"datasets":[{"logical_name":"libero_10","dataset_version":"1.0.0","path":"/data/libero_10"}]}}'\n''',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    result = _run("dataset", "list", environment={
        "PATH": f"{binaries}:{os.environ['PATH']}", "OVLAB_PYTHON": sys.executable,
        "OVLAB_FAKE_DOCKER_LOG": str(log), "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data"),
        "COLUMNS": "120",
    })
    assert result.returncode == 0
    assert "DATASET" in result.stdout and "VERSION" in result.stdout and "PATH" in result.stdout
    assert "libero_10" in result.stdout and "/data/libero_10" in result.stdout
    assert log.read_text().splitlines()[-3:] == ["dataset", "list", "--json"]


def test_version_short_alias_and_unified_run_entry_point():
    short = _run("-v")
    long = _run("--version")
    assert short.returncode == long.returncode == 0
    assert short.stdout == long.stdout
    help_output = _run("run", "--help")
    assert help_output.returncode == 0
    assert "EXPERIMENT" in help_output.stdout
    assert "Other operations" not in help_output.stdout
    assert _run("run", "train", "validate", "profile.yaml").returncode == 2


def test_dataset_list_preserves_container_failure_status(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\necho "image failure" >&2\nexit 6\n', encoding="utf-8")
    docker.chmod(0o755)
    result = _run("dataset", "list", environment={
        "PATH": f"{binaries}:{os.environ['PATH']}", "OVLAB_PYTHON": sys.executable,
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data"),
    })
    assert result.returncode == 6
    assert "image failure" in result.stderr
    assert result.stdout == ""


def test_grouped_training_report_uses_reporting_container(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    log = tmp_path / "docker-arguments.txt"
    docker.write_text(
        '#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n', encoding="utf-8"
    )
    docker.chmod(0o755)
    data = tmp_path / "data"
    (data / "runs").mkdir(parents=True)
    result = _run("train", "report", "run-1", "--json", environment={
        "PATH": f"{binaries}:{os.environ['PATH']}", "OVLAB_DATA_ROOT": str(data),
        "OVLAB_FAKE_DOCKER_LOG": str(log),
    })
    assert result.returncode == 0
    assert log.read_text().splitlines()[-4:] == ["train", "report", "run-1", "--json"]


def test_unified_report_parses_single_run_and_export_subcommands():
    direct = _parser().parse_args(["report", "run-123", "--json"])
    assert direct.command == "report" and direct.report_command == "auto"
    assert direct.reference == "run-123" and direct.json
    with pytest.raises(SystemExit):
        _parser().parse_args(["report", "export", "grouped", "comparison", "run-123", "run-456"])
    grouped = _parser().parse_args(["export", "grouped", "comparison", "run-123", "run-456"])
    assert grouped.command == "export" and grouped.export_command == "grouped"


def test_report_shortcut_uses_isolated_export_in_reporting_image(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    log = tmp_path / "docker-arguments.txt"
    docker.write_text(
        '#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n'
        "printf '%s\\n' '{\"status\":\"success\",\"result\":{\"output\":\"/exports/run-123\",\"integrity\":\"verified\"}}'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    data = tmp_path / "data"
    run = data / "runs/evaluate/run-123"
    run.mkdir(parents=True)
    (run / "manifest.completed.json").write_text("{}")
    env = {"PATH": f"{binaries}:{os.environ['PATH']}", "OVLAB_DATA_ROOT": str(data),
           "OVLAB_FAKE_DOCKER_LOG": str(log)}
    result = _run("report", "run-123", "--json", environment=env)
    assert result.returncode == 0, result.stderr
    exported = json.loads(result.stdout)["result"]["exports"][0]
    assert exported["type"] == "isolated" and exported["runs"] == ["run-123"]
    assert exported["output"] == "/exports/run-123"
    assert log.read_text().splitlines()[-7:] == ["export", "isolated", "--category", "evaluate", "--run", "run-123", "--json"]


def test_report_shortcut_dispatches_to_automatic_export(monkeypatch):
    import ovlab_benchctl.report_auto as report_auto
    expected = {"source_type": "run", "source": "run-123", "exports": []}
    monkeypatch.setattr(report_auto, "generate_report", lambda reference, **kwargs: expected)
    result, json_mode, _ = _dispatch(_parser().parse_args(["report", "run-123", "--json"]))
    assert result == expected and json_mode


def test_report_help_exposes_automatic_workflow_not_legacy_export_actions():
    result = _run("report", "--help")
    assert result.returncode == 0
    assert "RUN_OR_EXPERIMENT_ID" in result.stdout
    assert "profiles" in result.stdout and "verify" in result.stdout
    assert "generate" not in result.stdout and "publish" in result.stdout
    assert "report export" not in result.stdout


def test_report_publish_maps_machine_output_into_reporting_container(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "docker-arguments.txt"
    docker = binaries / "docker"
    docker.write_text(
        '#!/bin/sh\nif [ "${1:-}" = image ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$OVLAB_FAKE_DOCKER_LOG"\n', encoding="utf-8"
    )
    docker.chmod(0o755)
    data = tmp_path / "data"
    (data / "runs").mkdir(parents=True)
    env = {"PATH": f"{binaries}:{os.environ['PATH']}", "OVLAB_DATA_ROOT": str(data),
           "OVLAB_FAKE_DOCKER_LOG": str(log)}
    output = data / "exports" / "machine-report"
    result = _run("report", "publish", "run-1", "--output", str(output), environment=env)
    assert result.returncode == 0, result.stderr
    arguments = log.read_text().splitlines()
    assert arguments[arguments.index("--output") + 1] == "/var/lib/ovlab/exports/machine-report"
    invalid = _run("report", "publish", "run-1", "--output", str(tmp_path / "outside"), environment=env)
    assert invalid.returncode == 3 and "must be inside" in invalid.stderr
