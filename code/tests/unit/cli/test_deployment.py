"""Dependency-light tests for one-command Docker Compose orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ovlab_benchctl.deployment import ComposeDeployment, ComposeDeploymentError
from ovlab_benchctl.config_bundle import ConfigBundleBuilder


REPOSITORY = Path(__file__).resolve().parents[4]
OVLAB = REPOSITORY / "ovlab"
ENV_FILE = "deploy/compose/.env.example"
LORA = "configs/experiments/libero10-lora-merged-rpc-smoke.yaml"
OFT = "configs/experiments/libero10-openvla-oft-rpc-smoke.yaml"
_PREFLIGHT_IMAGES = ComposeDeployment._preflight_images


@dataclass
class _Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _Runner:
    def __init__(self, *returncodes: int, status_payload: str | None = None) -> None:
        self.returncodes = list(returncodes)
        self.calls: list[dict[str, object]] = []
        self.status_payload = status_payload

    def __call__(self, command, **kwargs):
        self.calls.append({"command": command, **kwargs})
        if command[-4:] == ["ps", "--all", "--format", "json"]:
            profile = command[command.index("--profile") + 1]
            policy = "policy-openvla-oft" if profile == "oft" else "policy-openvla"
            benchmark = "benchmark-oft" if profile == "oft" else "benchmark-openvla"
            payload = self.status_payload or "\n".join((
                json.dumps({"Service": policy, "State": "exited", "ExitCode": 0}),
                json.dumps({"Service": benchmark, "State": "exited", "ExitCode": 0}),
            ))
            return _Result(0, stdout=payload)
        result = _Result(self.returncodes.pop(0) if self.returncodes else 0)
        if result.returncode == 0 and "up" in command:
            runs = Path(kwargs["env"]["OVLAB_RUNS_ROOT"])
            (runs / "simulated-run").mkdir(exist_ok=True)
        return result


@pytest.fixture
def deployment_environment(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    return {
        "PATH": os.environ["PATH"],
        "OVLAB_DATASETS_PATH": str(datasets),
        "OVLAB_RUNS_ROOT": str(runs),
    }


@pytest.fixture(autouse=True)
def resolved_checkpoint(monkeypatch, tmp_path):
    snapshot = tmp_path / "resolved-checkpoint"
    snapshot.mkdir()
    spec = SimpleNamespace(
        resource_id="openvla-test", repo_id="owner/model", revision="a" * 40,
        expected_sha256="b" * 64,
    )
    checkpoint = SimpleNamespace(
        spec=spec, host_path=snapshot,
        container_path="/checkpoints/resolved/openvla-test", source_kind="test-cache",
        as_dict=lambda: {"checkpoint_id": "openvla-test", "host_path": str(snapshot)},
    )
    monkeypatch.setattr(ComposeDeployment, "_resolve_checkpoint", lambda self, plan: checkpoint)
    monkeypatch.setattr(ComposeDeployment, "_preflight_images", lambda self, plan: None)


def test_openvla_egl_run_preflights_starts_and_reaps_one_project(deployment_environment):
    runner = _Runner(0, 0, 0)
    deployment = ComposeDeployment(REPOSITORY, environment=deployment_environment, runner=runner)
    plan = deployment.plan(
        LORA,
        profile="openvla",
        renderer="egl",
        env_file=ENV_FILE,
        project_name="ovlab-test-openvla",
    )

    result = deployment.run(plan)

    assert result["status"] == result["cleanup"] == "completed"
    assert len(runner.calls) == 5
    config, up, status, down, publish = (call["command"] for call in runner.calls)
    assert config[-2:] == ["config", "--quiet"]
    assert "compose.glfw.yaml" not in " ".join(config)
    assert up[-6:] == [
        "up", "--pull", "never", "--no-build", "--exit-code-from", "benchmark-openvla",
    ]
    assert status[-4:] == ["ps", "--all", "--format", "json"]
    assert down[-3:] == ["down", "--volumes", "--remove-orphans"]
    assert publish[-6:] == ["--profile", "reporting", "run", "--rm", "--no-deps", "reporting"]
    assert runner.calls[-1]["env"]["OVLAB_REPORT_RUN_ID"] == "simulated-run"
    assert runner.calls[-1]["env"]["OVLAB_REPORT_ENABLED"] == "true"
    container_experiment = "/opt/ovlab/configs/experiments/libero10-lora-merged-rpc-smoke.yaml"
    assert all(
        call["env"]["OVLAB_EXPERIMENT_CONFIG"] == container_experiment
        for call in runner.calls
    )
    bundle_paths = {call["env"]["OVLAB_CONFIG_BUNDLE_PATH"] for call in runner.calls}
    assert len(bundle_paths) == 1
    assert not Path(next(iter(bundle_paths))).exists()
    assert all(
        call["env"]["OVLAB_CONFIG_BUNDLE_SHA256"]
        == result["config_bundle"]["bundle_sha256"]
        for call in runner.calls
    )
    assert all(call["env"]["OVLAB_RESOLVED_CHECKPOINT_ID"] == "openvla-test" for call in runner.calls)
    assert all(
        call["env"]["OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH"]
        == "/checkpoints/resolved/openvla-test"
        for call in runner.calls
    )


def test_success_reports_new_host_run_path(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    datasets = tmp_path / "datasets"
    datasets.mkdir()

    runner = _Runner(0, 0, 0, 0)
    deployment = ComposeDeployment(
        REPOSITORY,
        environment={
            "PATH": os.environ["PATH"],
            "OVLAB_DATASETS_PATH": str(datasets),
            "OVLAB_RUNS_ROOT": str(runs),
        },
        runner=runner,
    )
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-run-path",
    )

    result = deployment.run(plan)

    assert result["run_path"] == str((runs / "simulated-run").resolve())
    assert result["new_run_paths"] == [str((runs / "simulated-run").resolve())]
    assert result["postprocessing"]["canonical_run_modified"] is False


def test_reporting_failure_preserves_completed_canonical_run_and_is_not_silent(
    deployment_environment,
):
    runner = _Runner(0, 0, 0, 19)
    deployment = ComposeDeployment(
        REPOSITORY, environment=deployment_environment, runner=runner,
    )
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-reporting-failure",
    )

    with pytest.raises(ComposeDeploymentError, match="canonical benchmark run completed") as failure:
        deployment.run(plan)

    run_path = Path(deployment_environment["OVLAB_RUNS_ROOT"]) / "simulated-run"
    assert run_path.is_dir()
    assert str(run_path) in str(failure.value)
    assert runner.calls[-2]["command"][-3:] == ["down", "--volumes", "--remove-orphans"]
    assert runner.calls[-1]["command"][-1] == "reporting"


def test_completed_benchmark_without_a_unique_run_refuses_ambiguous_handoff(
    deployment_environment,
):
    class NoArtifactRunner(_Runner):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if result.returncode == 0 and "up" in command:
                (Path(kwargs["env"]["OVLAB_RUNS_ROOT"]) / "simulated-run").rmdir()
            return result

    runner = NoArtifactRunner(0, 0, 0)
    deployment = ComposeDeployment(
        REPOSITORY, environment=deployment_environment, runner=runner,
    )
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-no-run-handoff",
    )

    with pytest.raises(ComposeDeploymentError, match="exactly one canonical run"):
        deployment.run(plan)

    assert all(call["command"][-1] != "reporting" for call in runner.calls)


def test_oft_glfw_plan_selects_oft_services_and_renderer_overlay():
    deployment = ComposeDeployment(REPOSITORY)
    plan = deployment.plan(
        OFT,
        profile="oft",
        renderer="glfw",
        env_file=ENV_FILE,
        project_name="ovlab-test-oft",
    )

    assert plan.benchmark_service == "benchmark-oft"
    assert plan.compose_files[-1].endswith("deploy/compose/compose.glfw.yaml")
    assert plan.up_command[-2:] == ("--exit-code-from", "benchmark-oft")
    assert plan.experiment == OFT


def test_deployment_defaults_are_read_from_experiment_yaml():
    deployment = ComposeDeployment(REPOSITORY)

    lora = deployment.plan(
        LORA,
        env_file=ENV_FILE,
        project_name="ovlab-test-lora-defaults",
    )
    oft = deployment.plan(
        OFT,
        env_file=ENV_FILE,
        project_name="ovlab-test-oft-defaults",
    )

    assert (lora.profile, lora.renderer) == ("openvla", "egl")
    assert (oft.profile, oft.renderer) == ("oft", "egl")
    assert lora.profile_source == lora.renderer_source == "experiment"
    assert oft.profile_source == oft.renderer_source == "experiment"


def test_cli_renderer_override_takes_precedence_over_experiment():
    deployment = ComposeDeployment(REPOSITORY)
    plan = deployment.plan(
        LORA,
        renderer="glfw",
        env_file=ENV_FILE,
        project_name="ovlab-test-renderer-override",
    )

    assert plan.profile == "openvla"
    assert plan.renderer == "glfw"
    assert plan.profile_source == "experiment"
    assert plan.renderer_source == "cli"
    assert plan.compose_files[-1].endswith("deploy/compose/compose.glfw.yaml")


def test_incompatible_profile_override_is_rejected_before_docker():
    deployment = ComposeDeployment(REPOSITORY)
    with pytest.raises(ValueError, match="incompatible with policy type"):
        deployment.plan(OFT, profile="openvla", env_file=ENV_FILE)


def test_offline_flag_is_part_of_the_deployment_plan():
    deployment = ComposeDeployment(REPOSITORY)
    plan = deployment.plan(
        OFT,
        profile="oft",
        renderer="egl",
        env_file=ENV_FILE,
        project_name="ovlab-test-offline",
        offline=True,
    )

    assert plan.offline is True
    assert plan.document(side_effects_performed=False)["offline"] is True


def test_stale_policy_and_benchmark_images_fail_before_checkpoint_or_compose(
    deployment_environment,
):
    observed = []

    def inspect(command, **kwargs):
        observed.append((command, kwargs))
        return _Result(0, stdout="old-contract\n")

    deployment = ComposeDeployment(
        REPOSITORY, environment=deployment_environment, runner=inspect
    )
    plan = deployment.plan(
        OFT, profile="oft", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-stale-image",
    )

    with pytest.raises(ComposeDeploymentError, match="stale or incompatible") as failure:
        _PREFLIGHT_IMAGES(deployment, plan)

    assert "bash deploy/scripts/build-images.sh benchmark policy-openvla-oft reporting" in str(failure.value)
    assert len(observed) == 3
    assert all(call[0][:3] == ["docker", "image", "inspect"] for call in observed)


def test_openvla_preflight_requires_quantization_capable_policy_image(
    deployment_environment, monkeypatch,
):
    current_source = "a" * 64
    monkeypatch.setattr(
        ComposeDeployment, "_current_source_sha256", lambda self: current_source
    )

    def inspect(command, **kwargs):
        del kwargs
        image = command[-1]
        contract = (
            "resolved-checkpoint-config-bundle-v2"
            if image == "ovlab-benchmark-libero:local"
            else "canonical-readonly-reporting-v1"
            if image == "ovlab-reporting:local"
            else "resolved-checkpoint-v1"
        )
        return _Result(0, stdout=f"{contract}|{current_source}\n")

    deployment = ComposeDeployment(
        REPOSITORY, environment=deployment_environment, runner=inspect
    )
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-old-openvla-image",
    )

    with pytest.raises(ComposeDeploymentError, match="policy-openvla") as failure:
        _PREFLIGHT_IMAGES(deployment, plan)

    assert "build-images.sh policy-openvla" in str(failure.value)


def test_preflight_rejects_image_that_does_not_contain_current_source(monkeypatch):
    current_source = "a" * 64
    monkeypatch.setattr(
        ComposeDeployment, "_current_source_sha256", lambda self: current_source
    )

    def inspect(command, **kwargs):
        del kwargs
        image = command[-1]
        contract = (
            "resolved-checkpoint-quantization-config-bundle-v2"
            if image == "ovlab-policy-openvla:local"
            else "canonical-readonly-reporting-v1"
            if image == "ovlab-reporting:local"
            else "resolved-checkpoint-config-bundle-v2"
        )
        return _Result(0, stdout=f"{contract}|{'b' * 64}\n")

    deployment = ComposeDeployment(REPOSITORY, runner=inspect)
    plan = deployment.plan(
        LORA, env_file=ENV_FILE, project_name="ovlab-test-stale-source"
    )

    with pytest.raises(ComposeDeploymentError, match="source-manifest") as failure:
        _PREFLIGHT_IMAGES(deployment, plan)

    assert "build-images.sh benchmark policy-openvla reporting" in str(failure.value)


def test_failed_benchmark_is_cleaned_and_reported(deployment_environment):
    runner = _Runner(0, 17, 0)
    deployment = ComposeDeployment(REPOSITORY, environment=deployment_environment, runner=runner)
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-failure",
    )

    with pytest.raises(ComposeDeploymentError, match="exit code 17"):
        deployment.run(plan)

    assert runner.calls[-1]["command"][-3:] == ["down", "--volumes", "--remove-orphans"]


def test_policy_failure_is_reported_when_compose_up_returns_zero(deployment_environment):
    statuses = "\n".join((
        json.dumps({"Service": "policy-openvla-oft", "State": "exited", "ExitCode": 6}),
        json.dumps({"Service": "benchmark-oft", "State": "exited", "ExitCode": 0}),
    ))
    runner = _Runner(0, 0, 0, status_payload=statuses)
    deployment = ComposeDeployment(REPOSITORY, environment=deployment_environment, runner=runner)
    plan = deployment.plan(
        OFT, profile="oft", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-policy-failure",
    )

    with pytest.raises(ComposeDeploymentError, match="policy-openvla-oft exited with code 6"):
        deployment.run(plan)

    assert runner.calls[-1]["command"][-3:] == ["down", "--volumes", "--remove-orphans"]


def test_failed_compose_preflight_never_starts_services(deployment_environment):
    runner = _Runner(2)
    deployment = ComposeDeployment(REPOSITORY, environment=deployment_environment, runner=runner)
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-config-failure",
    )

    with pytest.raises(ComposeDeploymentError, match="configuration failed"):
        deployment.run(plan)

    assert len(runner.calls) == 1


def test_missing_dataset_mount_fails_before_compose(tmp_path):
    datasets = tmp_path / "missing-datasets"
    runs = tmp_path / "runs"
    runs.mkdir()
    runner = _Runner()
    deployment = ComposeDeployment(REPOSITORY, environment={
        "OVLAB_DATASETS_PATH": str(datasets),
        "OVLAB_RUNS_ROOT": str(runs),
    }, runner=runner)
    plan = deployment.plan(
        OFT, profile="oft", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-missing-cache",
    )

    with pytest.raises(ComposeDeploymentError, match="OVLAB_DATASETS_PATH"):
        deployment.run(plan)

    assert runner.calls == []


def test_unset_dataset_path_uses_managed_storage_next_to_runs(tmp_path):
    runs = tmp_path / "ovlab-data/runs"
    runs.mkdir(parents=True)
    runner = _Runner(0, 0, 0)
    deployment = ComposeDeployment(REPOSITORY, environment={
        "PATH": os.environ["PATH"],
        "OVLAB_RUNS_ROOT": str(runs),
    }, runner=runner)
    plan = deployment.plan(
        OFT, profile="oft", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-managed-datasets",
    )

    result = deployment.run(plan)

    datasets = tmp_path / "ovlab-data/datasets/libero"
    assert datasets.is_dir()
    assert result["datasets_root"] == str(datasets.resolve())
    assert all(
        call["env"]["OVLAB_DATASETS_PATH"] == str(datasets.resolve())
        for call in runner.calls
    )


def test_missing_runs_root_is_created_setgid_without_world_access(tmp_path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    runs = tmp_path / "data/runs"
    runner = _Runner(0, 0, 0)
    deployment = ComposeDeployment(REPOSITORY, environment={
        "OVLAB_DATASETS_PATH": str(datasets),
        "OVLAB_RUNS_ROOT": str(runs),
    }, runner=runner)
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-create-runs",
    )

    deployment.run(plan)

    assert runs.is_dir()
    assert runs.stat().st_mode & 0o7777 == 0o2770
    assert all(call["env"]["OVLAB_HOST_ARTIFACT_GID"] == str(os.getgid()) for call in runner.calls)


def test_dry_run_has_no_docker_side_effects():
    runner = _Runner()
    deployment = ComposeDeployment(REPOSITORY, runner=runner)
    plan = deployment.plan(
        LORA, profile="openvla", renderer="egl", env_file=ENV_FILE,
        project_name="ovlab-test-plan",
    )

    result = deployment.run(plan, dry_run=True)

    assert result["deployment"] == "docker-compose"
    assert result["side_effects_performed"] is False
    assert result["config_bundle"]["read_only"] is True
    assert result["config_bundle"]["file_count"] > 1
    assert runner.calls == []


def test_config_bundle_is_minimal_deterministic_and_read_only(tmp_path):
    builder = ConfigBundleBuilder(REPOSITORY)
    first = builder.build(LORA, "profiles/libero-bench-egl.yaml")
    second = builder.build(LORA, "profiles/libero-bench-egl.yaml")

    assert first == second
    paths = {item["path"] for item in first.files}
    assert "experiments/libero10-lora-merged-rpc-smoke.yaml" in paths
    assert "benchmarks/libero/libero10-smoke.yaml" in paths
    assert "benchmarks/libero/libero10.yaml" in paths
    assert "policies/openvla-lora/merged-libero10.yaml" in paths
    assert "resources/registry.yaml" in paths
    assert "profiles/libero-bench-egl.yaml" in paths
    assert "experiments/libero10-openvla-oft-rpc-smoke.yaml" not in paths

    with builder.materialize(first) as root:
        manifest = root / ".ovlab-bundle.json"
        assert manifest.is_file()
        assert json.loads(manifest.read_text(encoding="utf-8"))["bundle_sha256"] == first.sha256
        assert all((root / path).stat().st_mode & 0o222 == 0 for path in paths)
        materialized = root
    assert not materialized.exists()


def test_config_bundle_identity_changes_when_selected_yaml_changes(tmp_path):
    copied = tmp_path / "repository"
    copied.mkdir()
    source = REPOSITORY / "configs"
    destination = copied / "configs"
    shutil.copytree(source, destination)
    builder = ConfigBundleBuilder(copied)
    before = builder.build(LORA, "profiles/libero-bench-egl.yaml")
    experiment = destination / "experiments/libero10-lora-merged-rpc-smoke.yaml"
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace(
            "methodological reference", "methodological reference changed"
        ),
        encoding="utf-8",
    )
    after = builder.build(LORA, "profiles/libero-bench-egl.yaml")
    assert before.sha256 != after.sha256


def test_deployment_rejects_non_experiment_configuration():
    deployment = ComposeDeployment(REPOSITORY)
    with pytest.raises(ValueError, match="configs/experiments"):
        deployment.plan(
            "configs/policies/openvla-vanilla/base.yaml",
            profile="openvla",
            renderer="egl",
            env_file=ENV_FILE,
        )


def test_public_cli_exposes_compose_dry_run_without_invoking_docker():
    completed = subprocess.run(
        [
            str(OVLAB), "deploy", "run", LORA,
            "--profile", "openvla", "--renderer", "egl",
            "--env-file", ENV_FILE, "--project-name", "ovlab-cli-test",
            "--offline", "--dry-run", "--json",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0 and completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["command"] == "deploy run"
    assert payload["result"]["profile"] == "openvla"
    assert payload["result"]["offline"] is True
    assert payload["result"]["side_effects_performed"] is False


def test_public_cli_reads_profile_and_renderer_from_experiment():
    completed = subprocess.run(
        [
            str(OVLAB), "deploy", "run", OFT,
            "--env-file", ENV_FILE, "--project-name", "ovlab-cli-yaml-defaults",
            "--offline", "--dry-run", "--json",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0 and completed.stderr == ""
    result = json.loads(completed.stdout)["result"]
    assert result["profile"] == "oft"
    assert result["renderer"] == "egl"
    assert result["profile_source"] == "experiment"
    assert result["renderer_source"] == "experiment"


def test_deploy_dry_run_needs_only_system_python_and_standard_library():
    completed = subprocess.run(
        [
            "/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "HOME=/tmp",
            f"OVLAB_ROOT={REPOSITORY}", str(OVLAB), "deploy", "run", LORA,
            "--env-file", ENV_FILE, "--project-name", "ovlab-stdlib-test",
            "--dry-run", "--json",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0 and completed.stderr == ""
    assert json.loads(completed.stdout)["result"]["deployment"] == "docker-compose"
