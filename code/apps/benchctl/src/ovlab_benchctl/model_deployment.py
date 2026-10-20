"""Build and retain a policy-only Docker service, independently of evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .config_bundle import ConfigBundleBuilder
from .config_paths import resolve_config_path
from .errors import ConfigCompatibilityError, ConfigReferenceError
from .image_metadata import IMAGE_CONTRACT_LABEL
from .strict_yaml import load
from .training_identity import atomic_json


_ROLES = {
    "openvla_vanilla": "policy-openvla",
    "openvla_lora_merged": "policy-openvla",
    "openvla_oft": "policy-openvla-oft",
    "openvla_quic": "policy-openvla-quic",
}
_IMAGES = {role: f"ovlab-{role}:local" for role in set(_ROLES.values())}
_CONTRACT = "resolved-checkpoint-quantization-config-bundle-v2"


def _hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ModelDeploymentPlan:
    deployment_id: str
    target: str
    role: str
    image: str
    build: bool
    host: str
    port: int | None
    container_port: int
    restart: str
    readiness_timeout_s: int
    gpu: str
    quantization: str
    quantization_override: str | None
    offline: bool
    local_profile: str | None
    state_directory: str
    config_hash: str
    bundle_sha256: str

    @property
    def container_name(self):
        return f"ovlab-{self.deployment_id}"


class ModelDeployment:
    def __init__(self, repository_root: Path, *, environment=None, runner=None):
        self.repository_root = Path(repository_root).resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.runner = runner or subprocess.run
        data = self.environment.get("OVLAB_MODEL_DATA_ROOT", self.environment.get(
            "OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data"))
        self.data_root = Path(data).expanduser().resolve()
        self.state_root = self.data_root / "deployments"

    def plan(self, target, *, quantization=None, port=None, offline=False, local_profile=None):
        path = resolve_config_path(target, self.repository_root, "deploy")
        config_root = self.repository_root / "configs"
        if not path.is_relative_to(config_root / "deploy"):
            raise ConfigReferenceError("deployment configs must be below configs/deploy/")
        builder = ConfigBundleBuilder(self.repository_root)
        reference = path.relative_to(config_root).as_posix()
        doc = builder.load_component(reference, "deployment")
        policy = builder.load_component(doc["components"]["policy"], "policy")
        role = _ROLES.get(policy["type"])
        if role is None:
            raise ConfigCompatibilityError("deployment requires a production policy")
        if policy["settings"]["checkpoint_id"] != policy["settings"]["processor_id"]:
            raise ConfigCompatibilityError("model deployment requires a shared model and processor checkpoint")
        selected = quantization if quantization is not None else doc.get("runtime", {}).get("quantization")
        bundle = builder.build(f"configs/{reference}", None, quantization=selected, kind="deployment")
        effective = selected if selected is not None else policy["settings"]["runtime"]["quantization"]
        settings = doc.get("service", {})
        selected_port = port if port is not None else settings.get("port", 8000)
        if selected_port is not None and (type(selected_port) is not int or not 1 <= selected_port <= 65535):
            raise ConfigCompatibilityError("deployment port must be in 1..65535")
        profile = local_profile or self.environment.get("OVLAB_LOCAL_PROFILE")
        if profile:
            profile = Path(profile).expanduser()
            profile = profile if profile.is_absolute() else self.repository_root / profile
            if not profile.is_file():
                raise ConfigReferenceError(f"local profile does not exist: {profile}")
            profile = str(profile.resolve())
        normalized = {**doc, "effective_quantization": effective, "published_port": selected_port}
        return ModelDeploymentPlan(
            doc["id"], f"configs/{reference}", role, doc.get("image", _IMAGES[role]), doc.get("build", True),
            settings.get("host", "127.0.0.1"), selected_port, settings.get("container_port", 8000),
            settings.get("restart", "unless-stopped"), settings.get("readiness_timeout_s", 900),
            doc.get("runtime", {}).get("gpu", "all"), effective, selected, offline, profile,
            str(self.state_root / doc["id"]), _hash(normalized), bundle.sha256,
        )

    def _command(self, plan, checkpoint=None):
        checkpoint_id = checkpoint.spec.resource_id if checkpoint else "<checkpoint-id>"
        checkpoint_path = str(checkpoint.host_path) if checkpoint else "<resolved-checkpoint-path>"
        state = Path(plan.state_directory)
        port = str(plan.container_port)
        command = [
            "docker", "run", "--detach", "--name", plan.container_name,
            "--label", "ovlab.managed=model-deployment", "--label", f"ovlab.config-hash={plan.config_hash}",
            "--read-only", "--user", "10001:10001", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--restart", plan.restart,
            "--gpus", plan.gpu, "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m,uid=10001,gid=10001,mode=0700",
            "--mount", f"type=bind,source={state / 'configs'},target=/opt/ovlab/configs,readonly",
            "--mount", f"type=bind,source={checkpoint_path},target=/checkpoints/resolved/{checkpoint_id},readonly",
            "--health-cmd", f"ovlab service health --host 127.0.0.1 --port {port} --json",
            "--health-interval", "5s", "--health-timeout", "8s", "--health-start-period", f"{plan.readiness_timeout_s}s",
            "--health-retries", "3",
        ]
        if plan.port is not None:
            command += ["--publish", f"{plan.host}:{plan.port}:{plan.container_port}"]
        env = {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/tmp/huggingface",
            "HF_MODULES_CACHE": "/tmp/transformers-modules", "OVLAB_CONFIG_BUNDLE_SHA256": plan.bundle_sha256,
            "OVLAB_RESOLVED_CHECKPOINT_ID": checkpoint_id,
            "OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH": f"/checkpoints/resolved/{checkpoint_id}",
        }
        if checkpoint:
            env.update(OVLAB_CHECKPOINT_REVISION=checkpoint.spec.revision,
                       OVLAB_CHECKPOINT_SHA256=checkpoint.spec.expected_sha256)
        policy = ConfigBundleBuilder(self.repository_root).load_component(
            ConfigBundleBuilder(self.repository_root).load_component(
                plan.target.removeprefix("configs/"), "deployment")["components"]["policy"], "policy")
        if policy["type"] == "openvla_quic":
            adapter = self.environment.get("OVLAB_QUIC_ADAPTER_HOST_PATH", policy["settings"]["quic"]["adapter_path"])
            adapter_path = Path(adapter).expanduser()
            if not adapter_path.is_absolute():
                adapter_path = self.repository_root / adapter_path
            command += ["--mount", f"type=bind,source={adapter_path.resolve()},target=/checkpoints/quic-adapter,readonly"]
            env["OVLAB_QUIC_ADAPTER_PATH"] = "/checkpoints/quic-adapter"
        for key, value in env.items():
            command += ["--env", f"{key}={value}"]
        command += [plan.image, "service", "serve", f"/opt/ovlab/{plan.target}",
                    "--host", "0.0.0.0", "--port", port, "--persistent"]
        return command

    def _exec(self, command, *, capture=True):
        output = {"capture_output": True} if capture else {"stdout": sys.stderr, "stderr": sys.stderr}
        return self.runner(command, cwd=self.repository_root, env=self.environment,
                           text=True, check=False, **output)

    def _checked(self, command, *, capture=True):
        result = self._exec(command, capture=capture)
        if result.returncode:
            raise RuntimeError(f"deployment command failed ({command[:3]}): {getattr(result, 'stderr', '')}")
        return result

    def _container(self, name, *, required=True):
        result = self._exec(["docker", "container", "inspect", name])
        if result.returncode:
            if not required and "No such" in (result.stderr or ""):
                return None
            raise RuntimeError(f"cannot inspect deployment container {name}: {result.stderr}")
        return json.loads(result.stdout)[0]

    def _checkpoint(self, plan):
        from .checkpointing import (CheckpointResolver, LatestCheckpointSelection, checkpoint_spec,
                                    default_global_cache, local_checkpoint_override)
        spec = checkpoint_spec(self.repository_root, plan.target)
        resolver = CheckpointResolver(
            global_cache=Path(self.environment.get("OVLAB_GLOBAL_HF_CACHE", default_global_cache(self.environment))),
            managed_cache=self.data_root / "checkpoints/huggingface",
        )
        override = local_checkpoint_override(plan.local_profile, spec.resource_id)
        if isinstance(spec, LatestCheckpointSelection):
            if plan.offline and spec.mode == "remote/main":
                raise ConfigCompatibilityError("remote/main requires online deployment")
            return resolver.resolve_latest(spec.resource_id, spec.repo_id, local_path=override,
                                           mode=spec.mode, allow_download=not plan.offline)
        return resolver.resolve(spec, local_path=override, offline=plan.offline)

    def run(self, plan, *, dry_run=False):
        build_command = ["bash", str(self.repository_root / "deploy/scripts/build-images.sh"), plan.role]
        preview = {**asdict(plan), "container_name": plan.container_name, "build_command": build_command if plan.build else None,
                   "run_command": self._command(plan), "side_effects_performed": False}
        if dry_run:
            return preview
        with self._lock(plan.deployment_id):
            return self._start(plan, build_command)

    @contextmanager
    def _lock(self, identifier):
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / f".{identifier}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigCompatibilityError(f"deployment {identifier} is being changed by another command") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _start(self, plan, build_command):
        if self._container(plan.container_name, required=False) is not None:
            raise ConfigCompatibilityError(f"deployment {plan.deployment_id} already exists; stop it before redeploying")
        if plan.build:
            self._checked(build_command, capture=False)
            if plan.image != _IMAGES[plan.role]:
                self._checked(["docker", "tag", _IMAGES[plan.role], plan.image])
        image = json.loads(self._checked(["docker", "image", "inspect", plan.image]).stdout)[0]
        labels = image["Config"].get("Labels", {})
        if labels.get(IMAGE_CONTRACT_LABEL) != _CONTRACT or labels.get("io.github.ovlab.service.transport") != "tcp-persistent-v1":
            raise ConfigCompatibilityError("deployment image has an incompatible policy contract")
        checkpoint = self._checkpoint(plan)
        state = Path(plan.state_directory)
        state.mkdir(parents=True, exist_ok=True)
        state.chmod(0o755)
        if (state / "configs").exists():
            shutil.rmtree(state / "configs")
        builder = ConfigBundleBuilder(self.repository_root)
        bundle = builder.build(plan.target, None, quantization=plan.quantization_override, kind="deployment")
        if bundle.sha256 != plan.bundle_sha256:
            raise ConfigCompatibilityError("deployment configuration changed after planning")
        with builder.materialize(bundle) as root:
            shutil.copytree(root, state / "configs")
        manifest = {**asdict(plan), "container_name": plan.container_name, "image_id": image["Id"],
                    "status": "starting", "checkpoint": checkpoint.as_dict()}
        atomic_json(state / "deployment.json", manifest, exclusive=False, mode=0o644)
        created = False
        try:
            self._checked(self._command(plan, checkpoint))
            created = True
            deadline = time.monotonic() + plan.readiness_timeout_s
            while True:
                container = self._container(plan.container_name)
                status = container["State"]
                if status.get("Running") and status.get("Health", {}).get("Status") == "healthy":
                    break
                if not status.get("Running") or time.monotonic() >= deadline:
                    raise RuntimeError(f"deployment did not become ready: {plan.container_name}")
                time.sleep(1)
            manifest.update(status="ready", container_id=container["Id"])
            atomic_json(state / "deployment.json", manifest, exclusive=False, mode=0o644)
        except BaseException:
            if created:
                self._exec(["docker", "container", "rm", "--force", plan.container_name])
            manifest["status"] = "failed"
            atomic_json(state / "deployment.json", manifest, exclusive=False, mode=0o644)
            raise
        return {**manifest, "side_effects_performed": True}

    def _record(self, target):
        path = resolve_config_path(target, self.repository_root, "deploy")
        identifier = (ConfigBundleBuilder(self.repository_root).load_component(
            path.relative_to(self.repository_root / "configs").as_posix(), "deployment")["id"]
            if path.is_file() else str(target))
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", identifier) is None:
            raise ConfigReferenceError("deployment target must be a config name/path or deployment ID")
        record = self.state_root / identifier / "deployment.json"
        if not record.is_file():
            raise ConfigReferenceError(f"deployment not found: {identifier}")
        return record, json.loads(record.read_text())

    def inspect(self, target):
        _, record = self._record(target)
        container = self._container(record["container_name"])
        if container["Config"].get("Labels", {}).get("ovlab.managed") != "model-deployment":
            raise ConfigCompatibilityError("container is not an OVLAB model deployment")
        if container["Config"]["Labels"].get("ovlab.config-hash") != record["config_hash"]:
            raise ConfigCompatibilityError("deployment container configuration identity mismatch")
        if record.get("container_id") and container["Id"] != record["container_id"]:
            raise ConfigCompatibilityError("deployment container was replaced")
        return {**record, "container_id": container["Id"], "state": container["State"],
                "ports": container.get("NetworkSettings", {}).get("Ports", {})}

    def verify(self, target):
        record_path, record = self._record(target)
        inspected = self.inspect(target)
        if not inspected["state"].get("Running"):
            raise ConfigCompatibilityError("deployment container is not running")
        container = self._container(record["container_name"])
        if container["Image"] != record["image_id"] or container["Config"]["Labels"].get("ovlab.config-hash") != record["config_hash"]:
            raise ConfigCompatibilityError("deployment image or configuration identity differs from its record")
        bindings = container.get("HostConfig", {}).get("PortBindings", {}) or {}
        expected = {} if record["port"] is None else {
            f"{record['container_port']}/tcp": [{"HostIp": record["host"], "HostPort": str(record["port"])}]
        }
        if bindings != expected:
            raise ConfigCompatibilityError("deployment port bindings differ from its record")
        root = record_path.parent / "configs"
        bundle = json.loads((root / ".ovlab-bundle.json").read_text())
        digest = bundle.pop("bundle_sha256")
        if _hash(bundle) != digest or digest != record["bundle_sha256"]:
            raise ConfigCompatibilityError("deployment bundle identity mismatch")
        for item in bundle["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ConfigCompatibilityError("deployment bundle contains an invalid file")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ConfigCompatibilityError(f"deployment config file changed: {item['path']}")
        self._checked(["docker", "exec", record["container_name"], "ovlab", "service", "health",
                       "--host", "127.0.0.1", "--port", str(record["container_port"]), "--json"])
        return {"deployment_id": record["deployment_id"], "verified": True, "status": "ready", "inference_performed": False}

    def stop(self, target, *, dry_run=False):
        record_path, record = self._record(target)
        if dry_run:
            return self._stop(target, record_path, record, dry_run=True)
        with self._lock(record["deployment_id"]):
            record_path, record = self._record(target)
            return self._stop(target, record_path, record, dry_run=False)

    def _stop(self, target, record_path, record, *, dry_run):
        self.inspect(target)
        commands = [
            ["docker", "container", "stop", "--time", "30", record["container_name"]],
            ["docker", "container", "rm", record["container_name"]],
        ]
        if not dry_run:
            for command in commands:
                self._checked(command)
            record["status"] = "stopped"
            atomic_json(record_path, record, exclusive=False, mode=0o644)
        return {"deployment_id": record["deployment_id"], "commands": commands,
                "status": "planned" if dry_run else "stopped", "side_effects_performed": not dry_run}


def resolve_policy_service(application, path):
    """Construct policy settings without constructing a benchmark or renderer."""
    from types import SimpleNamespace
    from .schema import validate
    resolver = application.resolver
    doc = resolver.load_component(application._relative(path), "deployment")
    policy = resolver.load_component(doc["components"]["policy"], "policy")
    if "quantization" in doc.get("runtime", {}):
        policy["settings"]["runtime"]["quantization"] = doc["runtime"]["quantization"]
        validate(policy, doc["components"]["policy"], "policy")
    local = load(application._local_profile())
    validate(local, "local profile", "local_profile")
    registry = resolver.load_component(doc["resources"]["registry"], "resource_registry")
    env = application.environment
    checkpoints = {}
    for key, entry in registry["checkpoints"].items():
        resolved = {"source": entry["repo_id"], "revision": entry.get("revision"),
                    "expected_sha256": entry.get("expected_sha256")}
        if "artifact" in entry:
            resolved.update(artifact=entry["artifact"], method=entry["method"])
        override = local.get("resources", {}).get("checkpoints", {}).get(key, {}).get("local_path")
        if override:
            resolved["local_path"] = override
        if key == env.get("OVLAB_RESOLVED_CHECKPOINT_ID"):
            resolved.update(local_path=env["OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH"],
                            revision=env["OVLAB_CHECKPOINT_REVISION"], expected_sha256=env["OVLAB_CHECKPOINT_SHA256"])
        checkpoints[key] = resolved
    settings = policy["settings"]
    # Existing policy owners use this observation context to validate image shape
    # and artifact adaptation; no benchmark, task, protocol, or renderer is started.
    observation = {"height": doc.get("input", {}).get("height", 224),
                   "width": doc.get("input", {}).get("width", 224),
                   "profile": "native_oft" if policy["type"] == "openvla_oft" else "primary_rgb"}
    context = {"settings": {"suite": settings["unnorm_key"], "observation": observation}}
    action = resolver.load_component(settings["action"]["action_interface"], "action_interface")
    adapter = env.get("OVLAB_QUIC_ADAPTER_PATH", settings.get("quic", {}).get("adapter_path"))
    typed = resolver._policy(policy, context, resolver._action_spec(action), checkpoints, local["devices"], quic_adapter_path=adapter)
    return SimpleNamespace(policy_settings=typed, experiment_id=doc["id"], protocol_settings=SimpleNamespace(base_seed=0))
