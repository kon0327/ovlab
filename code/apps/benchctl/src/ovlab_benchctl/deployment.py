"""Dependency-light Docker Compose orchestration for the public OVLAB CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

from .schema import validate
from .strict_yaml import load
from .config_bundle import ConfigBundleBuilder


class ComposeDeploymentError(RuntimeError):
    """A Docker Compose deployment failed or could not be cleaned up."""


@dataclass(frozen=True)
class ComposeDeploymentPlan:
    experiment: str
    profile: str
    renderer: str
    profile_source: str
    renderer_source: str
    project_name: str
    env_file: str
    local_profile: str | None
    offline: bool
    benchmark_service: str
    reporting_enabled: bool
    reporting_profile: str
    compose_files: tuple[str, ...]
    config_command: tuple[str, ...]
    up_command: tuple[str, ...]
    status_command: tuple[str, ...]
    down_command: tuple[str, ...]
    postprocess_command: tuple[str, ...]

    def document(self, *, side_effects_performed: bool) -> dict[str, object]:
        return {
            "deployment": "docker-compose",
            "experiment": self.experiment,
            "profile": self.profile,
            "renderer": self.renderer,
            "profile_source": self.profile_source,
            "renderer_source": self.renderer_source,
            "project_name": self.project_name,
            "env_file": self.env_file,
            "local_profile": self.local_profile,
            "offline": self.offline,
            "benchmark_service": self.benchmark_service,
            "reporting_enabled": self.reporting_enabled,
            "reporting_profile": self.reporting_profile,
            "compose_files": list(self.compose_files),
            "config_command": list(self.config_command),
            "up_command": list(self.up_command),
            "status_command": list(self.status_command),
            "down_command": list(self.down_command),
            "postprocess_command": list(self.postprocess_command),
            "side_effects_performed": side_effects_performed,
        }


class ComposeDeployment:
    """Start and reap one isolated benchmark/policy Compose project."""

    _PROFILES = {
        "openvla": "benchmark-openvla",
        "oft": "benchmark-oft",
    }
    _POLICY_SERVICES = {
        "openvla": "policy-openvla",
        "oft": "policy-openvla-oft",
    }
    _RENDERERS = {"egl", "glfw"}
    _POLICY_PROFILES = {
        "openvla_vanilla": "openvla",
        "openvla_lora_merged": "openvla",
        "openvla_oft": "oft",
    }
    _RENDERER_EXECUTION_PROFILES = {
        "egl": "profiles/libero-bench-egl.yaml",
        "glfw": "profiles/libero-playground-glfw.yaml",
    }
    _IMAGE_CONTRACT_LABEL = "cz.cvut.ovlab.deployment.contract"
    _SOURCE_MANIFEST_LABEL = "cz.cvut.ovlab.source-manifest.sha256"
    _IMAGE_CONTRACTS = {
        "benchmark": "resolved-checkpoint-config-bundle-v2",
        "reporting": "canonical-readonly-reporting-v1",
        "policy-openvla": "resolved-checkpoint-quantization-config-bundle-v2",
        "policy-openvla-oft": "resolved-checkpoint-config-bundle-v2",
    }
    _PROFILE_IMAGES = {
        "openvla": (
            ("OVLAB_BENCHMARK_IMAGE", "ovlab-benchmark-libero:local", "benchmark"),
            ("OVLAB_OPENVLA_IMAGE", "ovlab-policy-openvla:local", "policy-openvla"),
            ("OVLAB_REPORTING_IMAGE", "ovlab-reporting:local", "reporting"),
        ),
        "oft": (
            ("OVLAB_BENCHMARK_IMAGE", "ovlab-benchmark-libero:local", "benchmark"),
            ("OVLAB_OFT_IMAGE", "ovlab-policy-openvla-oft:local", "policy-openvla-oft"),
            ("OVLAB_REPORTING_IMAGE", "ovlab-reporting:local", "reporting"),
        ),
    }

    def __init__(
        self,
        repository_root: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
        runner: Callable[..., object] = subprocess.run,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.runner = runner

    def _inside_repository(self, value: str | Path, *, kind: str, must_exist: bool = True) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if not path.is_relative_to(self.repository_root):
            raise ValueError(f"{kind} must be inside the OVLAB repository: {path}")
        if must_exist and not path.is_file():
            raise ValueError(f"{kind} does not exist: {path}")
        return path

    @staticmethod
    def _project_slug(experiment: str, profile: str, process_id: int) -> str:
        stem = Path(experiment).stem
        slug = re.sub(r"[^a-z0-9]+", "-", f"{stem}-{profile}".lower()).strip("-")
        return f"ovlab-{slug[:40]}-{process_id}"

    def plan(
        self,
        experiment: str | Path,
        *,
        profile: str | None = None,
        renderer: str | None = None,
        env_file: str | Path | None = None,
        local_profile: str | Path | None = None,
        offline: bool = False,
        project_name: str | None = None,
    ) -> ComposeDeploymentPlan:
        experiment_path = self._inside_repository(experiment, kind="experiment configuration")
        config_root = self.repository_root / "configs"
        if not experiment_path.is_relative_to(config_root / "experiments"):
            raise ValueError("Docker deployment requires an experiment below configs/experiments/")
        experiment_reference = str(experiment_path.relative_to(self.repository_root))

        experiment_doc = load(experiment_path)
        validate(experiment_doc, str(experiment_path), "experiment")
        configured = experiment_doc.get("deployment")
        if configured is None:
            raise ValueError(
                "experiment does not declare deployment.profile and deployment.renderer; "
                "add the portable deployment block"
            )
        selected_profile = profile or configured["profile"]
        selected_renderer = renderer or configured["renderer"]
        reporting = experiment_doc.get("reporting", {})
        reporting_enabled = reporting.get("enabled", True)
        reporting_profile = reporting.get("profile", "libero-task-default")
        if selected_profile not in self._PROFILES:
            raise ValueError(f"deployment profile must be one of: {', '.join(sorted(self._PROFILES))}")
        if selected_renderer not in self._RENDERERS:
            raise ValueError(f"renderer must be one of: {', '.join(sorted(self._RENDERERS))}")

        policy_reference = experiment_doc["components"]["policy"]
        policy_path = (config_root / policy_reference).resolve()
        if not policy_path.is_relative_to(config_root / "policies") or not policy_path.is_file():
            raise ValueError(f"experiment policy must resolve below configs/policies/: {policy_reference}")
        policy_type = load(policy_path).get("type")
        required_profile = self._POLICY_PROFILES.get(policy_type)
        if required_profile is None:
            raise ValueError(f"policy type {policy_type!r} has no Docker deployment profile")
        if selected_profile != required_profile:
            raise ValueError(
                f"deployment profile {selected_profile!r} is incompatible with policy type "
                f"{policy_type!r}; expected {required_profile!r}"
            )

        profile_source = "cli" if profile is not None else "experiment"
        renderer_source = "cli" if renderer is not None else "experiment"
        profile = selected_profile
        renderer = selected_renderer

        selected_env = env_file or self.repository_root / "deploy/compose/.env"
        env_path = self._inside_repository(selected_env, kind="Compose environment file")
        local_profile_path = None
        selected_local_profile = local_profile or self.environment.get("OVLAB_LOCAL_PROFILE")
        if selected_local_profile is not None:
            candidate = Path(selected_local_profile).expanduser()
            if not candidate.is_absolute():
                candidate = self.repository_root / candidate
            if not candidate.is_file():
                raise ValueError(f"local profile does not exist: {candidate.resolve()}")
            local_profile_path = str(candidate.resolve())
        base = self.repository_root / "deploy/compose/compose.yaml"
        files = [base]
        if renderer == "glfw":
            files.append(self.repository_root / "deploy/compose/compose.glfw.yaml")

        project = project_name or self._project_slug(experiment_reference, profile, os.getpid())
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
            raise ValueError("Compose project name must contain only lowercase letters, digits, '_' or '-'")

        common: list[str] = [
            "docker", "compose", "--env-file", str(env_path), "--project-name", project,
        ]
        for compose_file in files:
            common.extend(("--file", str(compose_file)))
        common.extend(("--profile", profile))
        benchmark = self._PROFILES[profile]
        return ComposeDeploymentPlan(
            experiment=experiment_reference,
            profile=profile,
            renderer=renderer,
            profile_source=profile_source,
            renderer_source=renderer_source,
            project_name=project,
            env_file=str(env_path),
            local_profile=local_profile_path,
            offline=offline,
            benchmark_service=benchmark,
            reporting_enabled=reporting_enabled,
            reporting_profile=reporting_profile,
            compose_files=tuple(str(path) for path in files),
            config_command=tuple((*common, "config", "--quiet")),
            up_command=tuple((
                *common, "up", "--pull", "never", "--no-build",
                "--exit-code-from", benchmark,
            )),
            status_command=tuple((*common, "ps", "--all", "--format", "json")),
            down_command=tuple((*common, "down", "--volumes", "--remove-orphans")),
            postprocess_command=tuple((
                *common, "--profile", "reporting", "run", "--rm", "--no-deps", "reporting",
            )),
        )

    def _execute(self, command: Sequence[str], environment: Mapping[str, str]):
        return self.runner(
            list(command),
            cwd=self.repository_root,
            env=dict(environment),
            text=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=False,
        )

    def _service_statuses(
        self,
        plan: ComposeDeploymentPlan,
        environment: Mapping[str, str],
    ) -> tuple[dict[str, object], ...]:
        completed = self.runner(
            list(plan.status_command),
            cwd=self.repository_root,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise ComposeDeploymentError(
                f"Docker Compose service-status inspection failed with exit code "
                f"{completed.returncode}"
            )
        try:
            rows = tuple(
                json.loads(line) for line in (completed.stdout or "").splitlines() if line.strip()
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ComposeDeploymentError("Docker Compose returned malformed service status") from exc
        services = {row.get("Service"): row for row in rows if isinstance(row, dict)}
        expected = (self._POLICY_SERVICES[plan.profile], plan.benchmark_service)
        missing = [service for service in expected if service not in services]
        if missing:
            raise ComposeDeploymentError(
                f"Docker Compose did not report required services: {', '.join(missing)}"
            )
        return tuple(
            {
                "Service": services[service].get("Service"),
                "State": services[service].get("State"),
                "Health": services[service].get("Health"),
                "ExitCode": services[service].get("ExitCode"),
            }
            for service in expected
        )

    @staticmethod
    def _status_failure(statuses: Sequence[Mapping[str, object]]) -> ComposeDeploymentError | None:
        failures = []
        for row in statuses:
            exit_code = row.get("ExitCode")
            if type(exit_code) is not int:
                failures.append(f"{row.get('Service', 'unknown')} has no integer exit code")
            elif exit_code != 0:
                failures.append(f"{row.get('Service', 'unknown')} exited with code {exit_code}")
        if failures:
            return ComposeDeploymentError("Docker Compose service failure: " + "; ".join(failures))
        return None

    def _environment_value(self, plan: ComposeDeploymentPlan, key: str) -> str | None:
        value = self.environment.get(key)
        if value is None:
            for raw_line in Path(plan.env_file).read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                candidate_key, candidate = line.split("=", 1)
                if candidate_key.strip() == key:
                    value = candidate.strip().strip("\"'")
                    break
        return value

    def _bind_source(self, plan: ComposeDeploymentPlan, key: str) -> Path | None:
        value = self._environment_value(plan, key)
        if not value or "${" in value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(plan.compose_files[0]).parent / path
        return path.resolve()

    def _preflight_bind_sources(self, plan: ComposeDeploymentPlan) -> None:
        runs = self._bind_source(plan, "OVLAB_RUNS_ROOT")
        if runs is None:
            raise ComposeDeploymentError(
                "OVLAB_RUNS_ROOT must resolve to a host directory"
            )
        if not runs.exists():
            runs.mkdir(mode=0o2770, parents=True)
            runs.chmod(0o2770)
        if not runs.is_dir():
            raise ComposeDeploymentError(f"OVLAB_RUNS_ROOT is not a directory: {runs}")
        derived = self._bind_source(plan, "OVLAB_DERIVED_ROOT") or (runs.parent / "derived")
        if not derived.exists():
            derived.mkdir(mode=0o2770, parents=True)
            derived.chmod(0o2770)
        if not derived.is_dir():
            raise ComposeDeploymentError(f"OVLAB_DERIVED_ROOT is not a directory: {derived}")
        exports = self._bind_source(plan, "OVLAB_EXPORTS_ROOT") or (runs.parent / "exports")
        if not exports.exists():
            exports.mkdir(mode=0o2770, parents=True)
            exports.chmod(0o2770)
        if not exports.is_dir():
            raise ComposeDeploymentError(f"OVLAB_EXPORTS_ROOT is not a directory: {exports}")

    def _current_source_sha256(self) -> str:
        source_manifest = runpy.run_path(
            str(self.repository_root / "deploy/scripts/source_manifest.py")
        )
        document = source_manifest["build_manifest"](self.repository_root)
        return str(document["source_content_sha256"])

    def _preflight_images(self, plan: ComposeDeploymentPlan) -> None:
        rebuild_roles = []
        expected_source_sha256 = self._current_source_sha256()
        for variable, default, role in self._PROFILE_IMAGES[plan.profile]:
            image = self._environment_value(plan, variable) or default
            expected_contract = self._IMAGE_CONTRACTS[role]
            completed = self.runner(
                [
                    "docker", "image", "inspect", "--format",
                    (
                        f'{{{{ index .Config.Labels "{self._IMAGE_CONTRACT_LABEL}" }}}}'
                        f'|{{{{ index .Config.Labels "{self._SOURCE_MANIFEST_LABEL}" }}}}'
                    ),
                    image,
                ],
                cwd=self.repository_root,
                env=dict(self.environment),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise ComposeDeploymentError(
                    f"required image is unavailable: {image}; build it with "
                    f"bash deploy/scripts/build-images.sh {role}"
                )
            labels = (completed.stdout or "").strip().split("|", 1)
            contract = labels[0]
            source_sha256 = labels[1] if len(labels) == 2 else ""
            if contract != expected_contract or source_sha256 != expected_source_sha256:
                rebuild_roles.append((
                    image,
                    role,
                    contract or "missing",
                    source_sha256 or "missing",
                ))
        if rebuild_roles:
            details = ", ".join(
                f"{image} ({self._IMAGE_CONTRACT_LABEL}={contract!r}, "
                f"{self._SOURCE_MANIFEST_LABEL}={source!r})"
                for image, _role, contract, source in rebuild_roles
            )
            roles = " ".join(role for _image, role, _contract, _source in rebuild_roles)
            raise ComposeDeploymentError(
                f"local deployment images are stale or incompatible: {details}; rebuild with "
                f"bash deploy/scripts/build-images.sh {roles}"
            )

    def _dataset_root(self, plan: ComposeDeploymentPlan) -> Path:
        explicit = self._environment_value(plan, "OVLAB_DATASETS_PATH")
        if explicit:
            path = Path(explicit).expanduser()
            if not path.is_absolute():
                path = Path(plan.compose_files[0]).parent / path
            resolved = path.resolve()
            if not resolved.is_dir():
                raise ComposeDeploymentError(
                    f"OVLAB_DATASETS_PATH must name an existing directory: {resolved}"
                )
            return resolved
        runs = self._bind_source(plan, "OVLAB_RUNS_ROOT")
        if runs is None:
            raise ComposeDeploymentError("cannot derive managed dataset storage without OVLAB_RUNS_ROOT")
        managed = (runs.parent / "datasets/libero").resolve()
        managed.mkdir(mode=0o755, parents=True, exist_ok=True)
        return managed

    def _managed_checkpoint_root(self, plan: ComposeDeploymentPlan) -> Path:
        explicit = self._environment_value(plan, "OVLAB_MANAGED_CHECKPOINTS_ROOT")
        if explicit:
            path = Path(explicit).expanduser()
            if not path.is_absolute():
                path = Path(plan.compose_files[0]).parent / path
            return path.resolve()
        runs = self._bind_source(plan, "OVLAB_RUNS_ROOT")
        if runs is None:
            raise ComposeDeploymentError("cannot derive managed checkpoint storage without OVLAB_RUNS_ROOT")
        return (runs.parent / "checkpoints/huggingface").resolve()

    def _resolve_checkpoint(self, plan: ComposeDeploymentPlan):
        from .checkpointing import (
            CheckpointResolver,
            checkpoint_spec,
            default_global_cache,
            local_checkpoint_override,
        )
        spec = checkpoint_spec(self.repository_root, plan.experiment)
        global_value = self._environment_value(plan, "OVLAB_GLOBAL_HF_CACHE")
        global_cache = (
            Path(global_value).expanduser().resolve()
            if global_value else default_global_cache(self.environment).resolve()
        )
        local_path = local_checkpoint_override(plan.local_profile, spec.resource_id)
        resolver = CheckpointResolver(
            global_cache=global_cache,
            managed_cache=self._managed_checkpoint_root(plan),
            progress=lambda message: print(
                f"ovlab: checkpoint: {message}", file=sys.stderr, flush=True
            ),
        )
        return resolver.resolve(spec, local_path=local_path, offline=plan.offline)

    @staticmethod
    def _run_directories(root: Path | None) -> set[Path]:
        if root is None or not root.is_dir():
            return set()
        return {path.resolve() for path in root.iterdir() if path.is_dir()}

    def run(self, plan: ComposeDeploymentPlan, *, dry_run: bool = False) -> dict[str, object]:
        bundle_builder = ConfigBundleBuilder(self.repository_root)
        bundle = bundle_builder.build(
            plan.experiment,
            self._RENDERER_EXECUTION_PROFILES[plan.renderer],
        )
        if dry_run:
            return {
                **plan.document(side_effects_performed=False),
                "config_bundle": bundle.summary(),
            }
        self._preflight_bind_sources(plan)
        datasets_root = self._dataset_root(plan)
        self._preflight_images(plan)
        checkpoint = self._resolve_checkpoint(plan)
        with bundle_builder.materialize(bundle) as bundle_root:
            environment = {
                **self.environment,
                "OVLAB_EXPERIMENT_CONFIG": (
                    f"/opt/ovlab/configs/{bundle.experiment}"
                ),
                "OVLAB_EXECUTION_PROFILE": (
                    f"/opt/ovlab/configs/{bundle.execution_profile}"
                ),
                "OVLAB_CONFIG_BUNDLE_PATH": str(bundle_root),
                "OVLAB_CONFIG_BUNDLE_SHA256": bundle.sha256,
                "OVLAB_DEPLOYMENT_PROFILE": plan.profile,
                "OVLAB_DATASETS_PATH": str(datasets_root),
                "OVLAB_HOST_ARTIFACT_GID": self.environment.get(
                    "OVLAB_HOST_ARTIFACT_GID", str(os.getgid())
                ),
                "OVLAB_RESOLVED_CHECKPOINT_ID": checkpoint.spec.resource_id,
                "OVLAB_RESOLVED_CHECKPOINT_PATH": str(checkpoint.host_path),
                "OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH": checkpoint.container_path,
                "OVLAB_CHECKPOINT_SOURCE_KIND": checkpoint.source_kind,
                "OVLAB_CHECKPOINT_REPOSITORY": checkpoint.spec.repo_id,
                "OVLAB_CHECKPOINT_REVISION": checkpoint.spec.revision,
                "OVLAB_CHECKPOINT_SHA256": checkpoint.spec.expected_sha256,
            }
            runs_root = self._bind_source(plan, "OVLAB_RUNS_ROOT")
            runs_before = self._run_directories(runs_root)
            checked = self._execute(plan.config_command, environment)
            if checked.returncode != 0:
                raise ComposeDeploymentError(
                    f"Docker Compose configuration failed with exit code {checked.returncode}"
                )

            run_error: ComposeDeploymentError | None = None
            service_statuses: tuple[dict[str, object], ...] = ()
            try:
                completed = self._execute(plan.up_command, environment)
                if completed.returncode != 0:
                    run_error = ComposeDeploymentError(
                        f"Docker Compose benchmark failed with exit code {completed.returncode}"
                    )
                try:
                    service_statuses = self._service_statuses(plan, environment)
                    status_error = self._status_failure(service_statuses)
                    if status_error is not None:
                        run_error = status_error if run_error is None else ComposeDeploymentError(
                            f"{run_error}; {status_error}"
                        )
                except ComposeDeploymentError as exc:
                    run_error = exc if run_error is None else ComposeDeploymentError(
                        f"{run_error}; {exc}"
                    )
            finally:
                cleaned = self._execute(plan.down_command, environment)
            if cleaned.returncode != 0:
                cleanup_error = ComposeDeploymentError(
                    f"Docker Compose cleanup failed with exit code {cleaned.returncode}"
                )
                if run_error is not None:
                    raise ComposeDeploymentError(f"{run_error}; {cleanup_error}") from run_error
                raise cleanup_error
            if run_error is not None:
                raise run_error
            new_runs = sorted(
                str(path) for path in self._run_directories(runs_root) - runs_before
            )
            if len(new_runs) != 1:
                raise ComposeDeploymentError(
                    "completed benchmark must create exactly one canonical run; "
                    f"observed {len(new_runs)} new run directories"
                )
            run_path = Path(new_runs[0])
            derived_root = self._bind_source(plan, "OVLAB_DERIVED_ROOT") or (run_path.parent.parent / "derived")
            exports_root = self._bind_source(plan, "OVLAB_EXPORTS_ROOT") or (run_path.parent.parent / "exports")
            postprocess_environment = {
                **environment,
                "OVLAB_REPORT_RUN_ID": run_path.name,
                "OVLAB_REPORT_PROFILE": plan.reporting_profile,
                "OVLAB_REPORT_ENABLED": "true" if plan.reporting_enabled else "false",
                "OVLAB_DERIVED_DISPLAY_ROOT": str(derived_root),
                "OVLAB_EXPORTS_DISPLAY_ROOT": str(exports_root),
            }
            published = self._execute(plan.postprocess_command, postprocess_environment)
            if published.returncode != 0:
                raise ComposeDeploymentError(
                    "canonical benchmark run completed and remains authoritative at "
                    f"{run_path}, but isolated reporting failed with exit code "
                    f"{published.returncode}"
                )
        result = {
            "deployment": "docker-compose",
            "experiment": plan.experiment,
            "profile": plan.profile,
            "renderer": plan.renderer,
            "project_name": plan.project_name,
            "status": "completed",
            "cleanup": "completed",
            "runs_root": None if runs_root is None else str(runs_root),
            "datasets_root": str(datasets_root),
            "new_run_paths": new_runs,
            "run_path": new_runs[0],
            "postprocessing": {
                "status": "completed",
                "service": "reporting",
                "run_id": Path(new_runs[0]).name,
                "report_enabled": plan.reporting_enabled,
                "report_profile": plan.reporting_profile,
                "canonical_run_modified": False,
            },
            "service_statuses": list(service_statuses),
            "checkpoint": checkpoint.as_dict(),
            "config_bundle": bundle.summary(),
            "side_effects_performed": True,
        }
        return result
