"""Read-only diagnostics for a host-side OVLAB deployment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import subprocess
import sys


class Doctor:
    """Collect independent, actionable checks without changing host state."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
        runner=subprocess.run,
        which=shutil.which,
    ) -> None:
        self.root = Path(repository_root).resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.runner = runner
        self.which = which
        self.checks: list[dict[str, str]] = []

    def _add(self, name: str, status: str, message: str, remedy: str = "") -> None:
        self.checks.append({
            "name": name, "status": status, "message": message, "remedy": remedy,
        })

    def _probe(self, command: list[str]):
        try:
            return self.runner(
                command, cwd=self.root, env=self.environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                timeout=5,
            ), None
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, str(exc)

    def _directory(self, name: str, path: Path, *, required: bool) -> None:
        if path.is_dir():
            if os.access(path, os.R_OK | os.W_OK | os.X_OK):
                self._add(name, "ok", f"accessible directory: {path}")
            else:
                self._add(name, "error", f"directory is not accessible for writing: {path}",
                          "Grant the current user read, write, and traverse access.")
        elif path.exists():
            self._add(name, "error", f"path is not a directory: {path}",
                      "Configure a directory at this path.")
        elif required:
            self._add(name, "error", f"required directory is missing: {path}",
                      "Create the directory or correct its path in the Compose environment.")
        else:
            parent = next((item for item in path.parents if item.exists()), None)
            if parent is None or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                self._add(name, "error", f"directory cannot be created under {path}",
                          "Choose a writable host path in the Compose environment.")
            else:
                self._add(name, "warning", f"directory is absent; deployment will create it: {path}")

    def run(self, *, experiment: str | None = None, env_file: str = "deploy/compose/.env") -> dict[str, object]:
        self.checks = []
        required = ("VERSION", "configs", "deploy/compose/compose.yaml")
        missing = [item for item in required if not (self.root / item).exists()]
        if missing:
            self._add("repository", "error", f"missing OVLAB files: {', '.join(missing)}",
                      "Set OVLAB_ROOT to a complete OVLAB checkout.")
        else:
            self._add("repository", "ok", f"OVLAB checkout: {self.root}")

        if sys.version_info >= (3, 10):
            self._add("python", "ok", f"Python {sys.version_info.major}.{sys.version_info.minor}")
        else:
            self._add("python", "error", "Python 3.10 or newer is required",
                      "Run OVLAB with Python 3.10 or newer.")

        selected_env = Path(env_file).expanduser()
        if not selected_env.is_absolute():
            selected_env = self.root / selected_env
        selected_env = selected_env.resolve()
        env_ready = selected_env.is_relative_to(self.root) and selected_env.is_file()
        if not selected_env.is_relative_to(self.root):
            self._add("compose-env", "error", f"Compose environment is outside the checkout: {selected_env}",
                      "Choose an environment file inside the OVLAB checkout.")
        elif not selected_env.is_file():
            self._add("compose-env", "error", f"Compose environment file is missing: {selected_env}",
                      "Copy deploy/compose/.env.example to deploy/compose/.env and set host paths.")
        else:
            self._add("compose-env", "ok", f"Compose environment file: {selected_env}")

        docker_ready = False
        if self.which("docker") is None:
            self._add("docker", "error", "Docker CLI is unavailable",
                      "Install Docker Engine and make the docker command available on PATH.")
            self._add("compose", "skipped", "Docker CLI is unavailable")
            self._add("daemon", "skipped", "Docker CLI is unavailable")
        else:
            compose, problem = self._probe(["docker", "compose", "version", "--short"])
            if problem or compose.returncode != 0:
                self._add("compose", "error", f"Docker Compose is unavailable: {problem or (compose.stderr or '').strip()}",
                          "Install the Docker Compose plugin and check Docker CLI configuration.")
            else:
                self._add("compose", "ok", f"Docker Compose {(compose.stdout or '').strip()}")
            daemon, problem = self._probe(["docker", "info", "--format", "{{.ServerVersion}}"])
            if problem or daemon.returncode != 0:
                self._add("daemon", "error", f"Docker daemon is unavailable: {problem or (daemon.stderr or '').strip()}",
                          "Start Docker Engine and grant this user access to its socket.")
            else:
                docker_ready = True
                self._add("daemon", "ok", f"Docker Engine {(daemon.stdout or '').strip()}")

        if experiment is not None:
            self._experiment(experiment, selected_env, env_ready, docker_ready)

        counts = Counter(check["status"] for check in self.checks)
        return {
            "schema_version": "ovlab-doctor/1.0.0",
            "status": "error" if counts["error"] else "warning" if counts["warning"] else "ok",
            "experiment": experiment,
            "checks": self.checks,
            "counts": {name: counts[name] for name in ("ok", "warning", "error", "skipped")},
        }

    def _experiment(self, experiment: str, env_file: Path, env_ready: bool, docker_ready: bool) -> None:
        if not env_ready:
            self._add("experiment", "skipped", "Compose environment file is unavailable")
            self._add("images", "skipped", "Experiment profile is unavailable")
            return
        from .config_bundle import ConfigBundleBuilder
        from .deployment import ComposeDeployment

        deployment = ComposeDeployment(self.root, environment=self.environment)
        try:
            plan = deployment.plan(experiment, env_file=env_file)
            bundle = ConfigBundleBuilder(self.root).build(
                plan.experiment, deployment._RENDERER_EXECUTION_PROFILES[plan.renderer],
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self._add("experiment", "error", f"experiment cannot be deployed: {exc}",
                      "Correct the experiment configuration and its referenced files.")
            self._add("images", "skipped", "Experiment profile is unavailable")
            return
        self._add("experiment", "ok", f"{plan.experiment}: {plan.profile}/{plan.renderer}; config bundle {bundle.sha256}")

        runs = deployment._bind_source(plan, "OVLAB_RUNS_ROOT")
        if runs is None:
            self._add("runs", "error", "OVLAB_RUNS_ROOT is unset or unresolved",
                      "Set OVLAB_RUNS_ROOT to a host directory in the Compose environment.")
        else:
            self._directory("runs", runs, required=False)
            exports = deployment._bind_source(plan, "OVLAB_EXPORTS_ROOT") or runs.parent / "exports"
            self._directory("exports", exports, required=False)
            dataset_value = deployment._environment_value(plan, "OVLAB_DATASETS_PATH")
            explicit_dataset = deployment._bind_source(plan, "OVLAB_DATASETS_PATH")
            if dataset_value and explicit_dataset is None:
                self._add("datasets", "error", "OVLAB_DATASETS_PATH is unresolved",
                          "Set OVLAB_DATASETS_PATH to an existing host directory.")
            elif explicit_dataset is not None:
                self._directory("datasets", explicit_dataset, required=True)
            else:
                self._directory("datasets", runs.parent / "datasets/libero", required=False)

        if not docker_ready:
            self._add("images", "skipped", "Docker daemon is unavailable")
            return
        for variable, default, role in deployment._PROFILE_IMAGES[plan.profile]:
            image = deployment._environment_value(plan, variable) or default
            expected = deployment._IMAGE_CONTRACTS[role]
            checked, problem = self._probe([
                "docker", "image", "inspect", "--format",
                f'{{{{ index .Config.Labels "{deployment._IMAGE_CONTRACT_LABEL}" }}}}', image,
            ])
            if problem or checked.returncode != 0:
                self._add(f"image:{role}", "error", f"image is unavailable: {image}",
                          f"Build it with: bash deploy/scripts/build-images.sh {role}")
            elif (checked.stdout or "").strip() != expected:
                self._add(f"image:{role}", "error", f"image {image} has a missing or incompatible deployment label",
                          f"Rebuild it with: bash deploy/scripts/build-images.sh {role}")
            else:
                self._add(f"image:{role}", "ok", f"compatible image: {image}")
