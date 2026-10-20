"""Policy-only deployment lifecycle and the public train/eval/deploy contract."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import ovlab_benchctl.cli as cli
from ovlab_benchctl.application import OvlabApplication
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.config_paths import resolve_config_path
from ovlab_benchctl.errors import ConfigCompatibilityError, ConfigSchemaError
from ovlab_benchctl.image_metadata import IMAGE_CONTRACT_LABEL
from ovlab_benchctl.model_deployment import ModelDeployment, resolve_policy_service
from ovlab_benchctl.strict_yaml import load

ROOT = Path(__file__).resolve().parents[4]


class Docker:
    def __init__(self):
        self.commands = []
        self.created = False
        self.running = True
        self.plan = None

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        value = ""
        if command[:3] == ["docker", "container", "inspect"]:
            if not self.created:
                return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such container")
            plan = self.plan
            value = json.dumps([{
                "Id": "container-1", "Image": "image-1",
                "Config": {"Labels": {"ovlab.managed": "model-deployment", "ovlab.config-hash": plan.config_hash}},
                "State": {"Running": self.running, "Health": {"Status": "healthy"}},
                "HostConfig": {"PortBindings": {f"{plan.container_port}/tcp": [{"HostIp": plan.host, "HostPort": str(plan.port)}]}},
            }])
        elif command[:3] == ["docker", "image", "inspect"]:
            value = json.dumps([{"Id": "image-1", "Config": {"Labels": {
                IMAGE_CONTRACT_LABEL: "resolved-checkpoint-quantization-config-bundle-v2",
                "io.github.ovlab.service.transport": "tcp-persistent-v1"}}}])
        elif command[:2] == ["docker", "run"]:
            self.created = True
            value = "container-1"
        elif command[:3] == ["docker", "container", "rm"]:
            self.created = False
        elif command[:2] == ["docker", "exec"]:
            value = '{"ready":true}'
        return SimpleNamespace(returncode=0, stdout=value, stderr="")


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    docker = Docker()
    owner = ModelDeployment(ROOT, environment={"OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data")}, runner=docker)
    spec = SimpleNamespace(resource_id="model", revision="a" * 40, expected_sha256="b" * 64)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(owner, "_checkpoint", lambda plan: SimpleNamespace(
        spec=spec, host_path=checkpoint, as_dict=lambda: {"resource_id": "model"}))
    return owner, docker


def test_model_deployment_builds_only_policy_and_retains_a_verified_service(deployment):
    owner, docker = deployment
    plan = owner.plan("openvla-libero10", quantization="4bit", port=8123)
    docker.plan = plan
    result = owner.run(plan)
    assert result["status"] == "ready"
    builds = [command for command in docker.commands if command[0] == "bash"]
    assert len(builds) == 1 and builds[0][-1] == "policy-openvla"
    command = next(command for command in docker.commands if command[:2] == ["docker", "run"])
    assert "127.0.0.1:8123:8000" in command
    assert command[-1] == "--persistent"
    assert not any("benchmark" in arg or arg == "reporting" for arg in command)
    assert docker.created
    assert owner.inspect("openvla-libero10")["state"]["Running"]
    assert owner.verify("openvla-libero10")["verified"] is True
    preview = owner.stop("openvla-libero10", dry_run=True)
    assert preview["side_effects_performed"] is False and docker.created
    assert owner.stop("openvla-libero10")["status"] == "stopped" and not docker.created


def test_model_deployment_dry_run_has_no_docker_or_disk_side_effects(deployment):
    owner, docker = deployment
    plan = owner.plan("openvla-libero10", quantization="8bit")
    result = owner.run(plan, dry_run=True)
    assert result["quantization"] == "8bit"
    assert result["side_effects_performed"] is False
    assert not docker.commands and not owner.state_root.exists()


def test_deployment_rejects_existing_container_and_cleans_failed_start(deployment):
    owner, docker = deployment
    plan = owner.plan("openvla-libero10")
    docker.plan = plan
    docker.created = True
    with pytest.raises(ConfigCompatibilityError, match="already exists"):
        owner.run(plan)
    docker.created = False
    docker.running = False
    with pytest.raises(RuntimeError, match="did not become ready"):
        owner.run(plan)
    assert not docker.created
    assert json.loads((Path(plan.state_directory) / "deployment.json").read_text())["status"] == "failed"


def test_deployment_verify_detects_modified_bundle(deployment):
    owner, docker = deployment
    plan = owner.plan("openvla-libero10")
    docker.plan = plan
    owner.run(plan)
    path = Path(plan.state_directory) / "configs/policies/openvla-lora/merged-libero10.yaml"
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n# modified\n")
    with pytest.raises(ConfigCompatibilityError, match="config file changed"):
        owner.verify("openvla-libero10")


def test_deployment_cli_override_reaches_service_settings_without_benchmark(tmp_path, monkeypatch):
    builder = ConfigBundleBuilder(ROOT)
    bundle = builder.build("configs/deploy/openvla-libero10.yaml", None, quantization="4bit", kind="deployment")
    repo = tmp_path / "repository"
    with builder.materialize(bundle) as root:
        shutil.copytree(root, repo / "configs")
    local = ROOT / "configs/local/gate-b-showrack.yaml"
    app = OvlabApplication(repo, environment={"OVLAB_LOCAL_PROFILE": str(local)})
    monkeypatch.setattr(app.resolver, "_benchmark", lambda *args: pytest.fail("deployment constructed a benchmark"))
    resolved = resolve_policy_service(app, repo / "configs/deploy/openvla-libero10.yaml")
    assert resolved.policy_settings.quantization.value == "4bit"
    assert load(ROOT / "configs/deploy/openvla-libero10.yaml")["runtime"]["quantization"] == "none"
    assert not any(item["path"].startswith(("benchmarks/", "protocols/", "metrics/")) for item in bundle.files)


@pytest.mark.parametrize("domain", ("train", "eval", "deploy"))
def test_domains_share_run_inspect_verify_target_and_dry_run(domain):
    parser = cli._parser()
    for operation in ("run", "inspect", "verify"):
        args = parser.parse_args([domain, operation, "example"])
        assert args.target == "example"
    assert parser.parse_args([domain, "run", "example", "--dry-run"]).dry_run


def test_train_dry_run_uses_planner_without_starting_trainer(monkeypatch):
    app = SimpleNamespace(train_plan=lambda target, **kwargs: {"target": target, "model_initialized": False})
    monkeypatch.setattr(cli, "_application", lambda: app)
    result, _, _ = cli._dispatch(cli._parser().parse_args(["train", "run", "openvla-full-reference", "--dry-run"]))
    assert result["side_effects_performed"] is False


def test_predefined_paths_cover_moved_training_and_evaluation():
    assert resolve_config_path("libero10-lora-merged-rpc-smoke", ROOT, "experiments/evaluation").is_file()
    assert resolve_config_path("quic/libero10-a1-qp2", ROOT, "experiments/training").is_file()
    assert not (ROOT / "configs/training").exists()
    assert not list((ROOT / "configs/experiments").glob("*.yaml"))
