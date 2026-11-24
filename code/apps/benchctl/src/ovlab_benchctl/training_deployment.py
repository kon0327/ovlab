"""Host owner for isolated dataset acquisition, training, and finalization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from ovlab_core.experiment_scope import ExperimentScope

from .policy_plugins import with_registry
from .application import OvlabApplication
from .datasets import DatasetRequest, DatasetStore, LiberoDatasetBridge
from .image_metadata import IMAGE_CONTRACT_LABEL
from .strict_yaml import load
from .training_errors import DatasetUnavailableError, TrainingRuntimeError
from .training_profiles import TrainingProfile, TrainingPlanner
from .training_runs import TrainingRunStore
from .training_identity import atomic_json, identity
from .versioning import repository_revision
from .wandb_tracking import track_run


class TrainingDeployment:
    """The only production owner allowed to start a training container."""

    def __init__(self, repository_root: Path, *, environment=None, runner=None):
        self.repository_root = repository_root.resolve()
        self.environment = dict(os.environ if environment is None else environment)
        self.runner = runner or subprocess.run
        self.model_data_root = Path(
            self.environment.get(
                "OVLAB_MODEL_DATA_ROOT",
                self.environment.get("OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data"),
            )
        ).expanduser().resolve()
        self.runs_root = Path(
            self.environment.get("OVLAB_RUNS_ROOT", self.model_data_root / "runs")
        ).expanduser().resolve()

    @with_registry
    def _profile(self, reference: str | Path) -> tuple[Path, TrainingProfile]:
        from .config_paths import resolve_config_path

        path = resolve_config_path(reference, self.repository_root, "experiments/training")
        if not path.is_file():
            raise TrainingRuntimeError(f"training profile does not exist: {path.resolve()}")
        from .training_ablations import load_training_profile
        return path.resolve(), load_training_profile(path.resolve(), config_root=(self.repository_root / "experiments" if path.resolve().is_relative_to(self.repository_root / "experiments") else self.repository_root / "configs"))

    def _gpu(self) -> tuple[int, float]:
        completed = self.runner(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            return 0, 0.0
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return len(values), max((float(value) / 1024.0 for value in values), default=0.0)

    def _docker_image(self, variable: str, default: str, expected_contract: str, *, plugin_role=None) -> tuple[str, str]:
        image = self.environment.get(variable, default)
        completed = self.runner(
            ["docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}|{{.Id}}"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise TrainingRuntimeError(f"required image is unavailable: {image}")
        labels_payload, digest = completed.stdout.strip().split("|", 1)
        labels = json.loads(labels_payload)
        if labels.get(IMAGE_CONTRACT_LABEL) != expected_contract:
            raise TrainingRuntimeError(
                f"image {image} has deployment contract {labels.get(IMAGE_CONTRACT_LABEL)!r}, "
                f"expected {expected_contract!r}"
            )
        if plugin_role is not None:
            from .plugin_identity import verify_plugin_image
            try:
                verify_plugin_image(labels, self.repository_root, plugin_role)
            except ValueError as exc:
                raise TrainingRuntimeError(f"image {image} differs from registered training plugin; rebuild {plugin_role}: {exc}") from exc
        return image, digest

    @staticmethod
    def _container_groups(*directories: Path) -> list[str]:
        """Grant the worker the actual groups of its managed bind mounts."""
        groups = {os.getgid(), *(path.stat().st_gid for path in directories)}
        return [argument for group in sorted(groups) for argument in ("--group-add", str(group))]

    @staticmethod
    def _prepare_scoped_hierarchy(base: Path, scope: ExperimentScope, category: str) -> Path:
        """Create each scoped parent with worker group access when host-owned."""
        current = base
        for part in (scope.experiment_id, category, *scope.node_parts):
            current = current / part
            if current.is_symlink():
                raise TrainingRuntimeError(f"experiment path contains a symbolic link: {current}")
            current.mkdir(exist_ok=True)
            if current.stat().st_uid == os.getuid():
                current.chmod(0o2770)
        return current

    def _prepare_base_checkpoint(self, plan: dict[str, object]) -> dict[str, object]:
        """Materialize HF blob links for an isolated, offline snapshot mount."""
        from .checkpointing import (
            CheckpointResolver, checkpoint_spec_by_id, checkpoint_spec_from_snapshot,
            default_global_cache,
        )

        execution = dict(plan["execution"])
        base_identity = plan["scientific"]["base_checkpoint"]
        base_path = Path(execution["model_host_path"])
        if base_identity.get("revision_selection") in {"local/main", "remote/main", "latest"}:
            spec = checkpoint_spec_from_snapshot(
                base_identity["resource_id"], base_identity["repository"],
                base_identity["revision"], base_path,
            )
            if spec.expected_sha256 != base_identity["aggregate_sha256"]:
                raise TrainingRuntimeError("selected latest checkpoint changed after planning")
        else:
            spec = checkpoint_spec_by_id(self.repository_root, base_identity["resource_id"])
            from .checkpointing import LatestCheckpointSelection
            if isinstance(spec, LatestCheckpointSelection):
                spec = checkpoint_spec_from_snapshot(
                    base_identity["resource_id"], base_identity["repository"],
                    base_identity["revision"], base_path,
                )
                if spec.expected_sha256 != base_identity["aggregate_sha256"]:
                    raise TrainingRuntimeError("selected checkpoint changed after planning")
        global_cache = Path(self.environment.get("OVLAB_GLOBAL_HF_CACHE", default_global_cache(self.environment)))
        resolved = CheckpointResolver(
            global_cache=global_cache,
            managed_cache=self.model_data_root / "checkpoints/huggingface",
            progress=lambda message: print(f"ovlab: checkpoint: {message}", file=sys.stderr, flush=True),
        ).resolve(spec, local_path=base_path, offline=True)
        execution.update(model_host_path=str(resolved.host_path), model_source_kind=resolved.source_kind)
        return {**plan, "execution": execution, "execution_plan_id": identity("training-execution", execution, 32)}

    def _explicit_dataset_acquisition(self, profile: TrainingProfile, dataset_image: str) -> None:
        dataset = profile.document["dataset"]
        assert isinstance(dataset, dict)
        reference = str(dataset["ref"])
        if reference.startswith("dataset-"):
            raise DatasetUnavailableError(f"immutable dataset {reference} is missing and cannot be inferred")
        provider, name = reference.split("/", 1)
        if provider != "libero":
            raise DatasetUnavailableError("automatic training acquisition supports only known LIBERO selectors")
        datasets_root = self.model_data_root / "datasets"
        datasets_root.mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "run", "--rm", "--network", "bridge", "--read-only",
            "--user", "10001:10001", *self._container_groups(datasets_root),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_MODEL_DATA_ROOT=/var/lib/ovlab/model-data",
            "--mount", f"type=bind,source={datasets_root},target=/var/lib/ovlab/model-data/datasets",
            dataset_image, "dataset", "fetch", "--source", "libero", "--name", name, "--json",
        ]
        completed = self.runner(command, check=False, stdout=sys.stderr)
        if completed.returncode != 0:
            raise DatasetUnavailableError(f"explicit LIBERO acquisition container failed with exit code {completed.returncode}")

    @with_registry
    def run(self, profile_reference: str | Path, *, allow_dataset_download: bool = False,
            fit_action_normalization: bool = False, interactive_progress: bool = False) -> dict[str, object]:
        scope = ExperimentScope.from_environment(self.environment)
        training_root = scope.runs_root(self.runs_root, "train") if scope is not None else self.runs_root / "train"
        for directory in (
            self.model_data_root / "datasets",
            self.model_data_root / "checkpoints",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o2770)
        checkpoint_local = self.model_data_root / "checkpoints" / "local"
        checkpoint_local.mkdir(parents=True, exist_ok=True)
        if checkpoint_local.stat().st_uid == os.getuid():
            checkpoint_local.chmod(0o2770)
        (self.model_data_root / "checkpoints" / "others").mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if scope is not None:
            self._prepare_scoped_hierarchy(self.runs_root, scope, "train")
        else:
            training_root.mkdir(parents=True, exist_ok=True)
            training_root.chmod(0o2770)
        profile_path, profile = self._profile(profile_reference)
        from .training_profiles import with_fitted_action_normalization
        profile = with_fitted_action_normalization(profile, fit_action_normalization)
        from .training_plugins import training_image, finalizer_image
        selected_image = training_image(profile.document, self.repository_root)
        training_image, training_digest = self._docker_image(
            selected_image["image_env"], selected_image["image"],
            "offline-dataset-ro-base-ro-training-run-rw-v1",
            plugin_role=selected_image["role"],
        )
        dataset_image, dataset_digest = self._docker_image(
            "OVLAB_DATASET_IMAGE", "ovlab-dataset:local", "dataset-write-only-or-finalizer-v1"
        )
        selected_finalizer = finalizer_image(profile.document, self.repository_root)
        finalizer_image_reference, finalizer_digest = self._docker_image(
            selected_finalizer["image_env"], selected_finalizer["image"],
            "dataset-write-only-or-finalizer-v1",
            plugin_role=selected_finalizer["role"],
        )
        reporting_image, reporting_digest = self._docker_image(
            "OVLAB_REPORTING_IMAGE", "ovlab-reporting:local", "canonical-readonly-reporting-v2"
        )
        datasets = DatasetStore(self.model_data_root)
        dataset_profile = profile.document["dataset"]
        assert isinstance(dataset_profile, dict)
        reference = str(dataset_profile["ref"])
        if not reference.startswith("dataset-"):
            provider, name = reference.split("/", 1)
            resolution = datasets.resolve(DatasetRequest(source=provider, name=name))
            if datasets.find_resolution(resolution.resolution_id) is None:
                if not allow_dataset_download:
                    raise DatasetUnavailableError(
                        f"dataset {reference!r} is missing; fetch it explicitly or repeat train run with --allow-dataset-download"
                    )
                self._explicit_dataset_acquisition(profile, dataset_digest)
        gpu_count, vram_gib = self._gpu() if selected_image.get("gpu", True) else (0, 0.0)
        plan = TrainingPlanner(self.repository_root, self.model_data_root).plan(
            profile,
            available_gpu_count=gpu_count,
            available_vram_gib=vram_gib,
            image_identity=training_digest,
            allow_checkpoint_download=True,
        )
        plan = self._prepare_base_checkpoint(plan)
        store = TrainingRunStore(self.model_data_root, runs_root=self.runs_root, scope=scope)
        context = store.create(
            load(profile_path),
            profile.document,
            plan,
            {
                "repository_revision": repository_revision(self.repository_root) or "unavailable",
                "source_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=self.repository_root, capture_output=True, text=True).stdout),
                "training_image": {"reference": training_image, "digest": training_digest},
                "finalizer_image": {"reference": finalizer_image_reference, "digest": finalizer_digest},
                "reporting_image": {"reference": reporting_image, "digest": reporting_digest},
                "gpu_count": gpu_count,
                "detected_vram_gib": vram_gib,
                "network": "disabled",
                "profile_source": str(profile_path),
                "fit_action_normalization_requested": fit_action_normalization,
            },
        )
        scientific = plan["scientific"]
        execution = plan["execution"]
        assert isinstance(scientific, dict) and isinstance(execution, dict)
        dataset_identity = scientific["dataset"]
        assert isinstance(dataset_identity, dict)
        dataset_manifest = datasets.inspect(str(dataset_identity["dataset_id"]))
        from .training_plugins import backend_for_profile
        from .plugin_mounts import validate_runtime_mounts, docker_bind
        resources = backend_for_profile(profile.document, self.repository_root).container_resources(plan, dataset_manifest)
        if not isinstance(resources, dict) or set(resources) - {"environment", "mounts"}:
            raise TrainingRuntimeError("training plugin must return environment and read-only mounts")
        worker_environment = resources.get("environment", {})
        if not isinstance(worker_environment, dict) or any(
            not isinstance(key, str) or re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) is None
            or not isinstance(value, str) or "\0" in value
            or key in {"OVLAB_PLUGIN_REGISTRY", "OVLAB_LOCAL_PROFILE", "OVLAB_ROOT", "OVLAB_TRAINING_BASE_CHECKPOINT"}
            for key, value in worker_environment.items()
        ):
            raise TrainingRuntimeError("training plugin returned invalid or reserved environment variables")
        worker_mounts = validate_runtime_mounts(resources.get("mounts", []))
        base_path = Path(str(execution["model_host_path"]))
        container_name = f"ovlab-training-{uuid.uuid4().hex[:12]}"
        # Workers derive the canonical run ID from the mounted root's basename.
        container_run_root = f"/var/lib/ovlab/training-run/{context.run_id}"
        command = [
            "docker", "run", *(["--tty"] if interactive_progress else []),
            "--name", container_name, "--rm", "--network", "none", "--read-only",
            *(["--gpus", f"device={self.environment.get('OVLAB_GPU_DEVICE', '0')}"] if selected_image.get("gpu", True) else []),
            "--user", "10001:10001", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            *self._container_groups(context.root),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_TRAINING_RUNTIME=isolated-container", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "WANDB_MODE=offline",
            "--env", f"OVLAB_PROGRESS={self.environment.get('OVLAB_PROGRESS', 'auto')}",
            *(["--env", f"OVLAB_TERMINAL_COLUMNS={shutil.get_terminal_size((80, 24)).columns or 80}"] if interactive_progress else []),
            "--env", "OVLAB_TRAINING_BASE_CHECKPOINT=/checkpoints/base",
            *[argument for key, value in worker_environment.items() for argument in ("--env", f"{key}={value}")],
            "--mount", f"type=bind,source={base_path},target=/checkpoints/base,readonly",
            *[argument for mount in worker_mounts for argument in ("--mount", docker_bind(mount["source"], mount["target"]))],
            "--mount", f"type=bind,source={context.root},target={container_run_root}",
            "--entrypoint", "python", training_digest,
            "-m", "ovlab_benchctl.training_worker", "train", "--run-root", container_run_root,
        ]
        atomic_json(context.root / "container-commands.json", {"training": command}, mode=0o644)
        try:
            completed = self.runner(command, check=False, stdout=sys.stderr)
        except KeyboardInterrupt as exc:
            self.runner(["docker", "stop", "--time", "20", container_name], check=False)
            self.runner(["docker", "rm", "--force", container_name], check=False)
            store.fail(context, exc, interrupted=True)
            raise
        if completed.returncode != 0:
            error = TrainingRuntimeError(f"isolated trainer failed with exit code {completed.returncode}; evidence: {context.root}")
            if not (context.root / "result.json").is_file():
                store.fail(context, error)
            raise error
        finalizer = [
            "docker", "run", "--rm", "--network", "none", "--read-only", "--user", "10001:10001",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            *self._container_groups(context.root, self.model_data_root / "checkpoints"),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=0700",
            "--mount", f"type=bind,source={context.root},target={container_run_root}",
            "--mount", f"type=bind,source={self.model_data_root / 'checkpoints'},target=/var/lib/ovlab/model-data/checkpoints",
            "--entrypoint", "python", finalizer_digest,
            "-m", "ovlab_benchctl.training_worker", "finalize", "--run-root", container_run_root,
            "--model-data-root", "/var/lib/ovlab/model-data",
        ]
        atomic_json(context.root / "container-commands.json", {"training": command, "finalizer": finalizer}, exclusive=False, mode=0o644)
        finalized = self.runner(finalizer, check=False, stdout=sys.stderr)
        if finalized.returncode != 0:
            raise TrainingRuntimeError(f"checkpoint finalizer failed; staged training evidence remains at {context.root}")
        tracking = track_run("train", context.root, self.environment)
        exports_root = Path(
            self.environment.get("OVLAB_EXPORTS_ROOT", self.model_data_root / "exports")
        ).expanduser().resolve()
        exports_root.mkdir(parents=True, exist_ok=True)
        exports_root.chmod(0o2770)
        export_node_root = scope.exports_root(exports_root, "train") if scope is not None else exports_root / "train"
        interactive_root = export_node_root / "interactive"
        if scope is not None:
            self._prepare_scoped_hierarchy(exports_root, scope, "train")
        else:
            export_node_root.mkdir(parents=True, exist_ok=True)
            export_node_root.chmod(0o2770)
        interactive_root.mkdir(exist_ok=True)
        if interactive_root.stat().st_uid == os.getuid():
            interactive_root.chmod(0o2770)
        scope_environment = (["--env", f"OVLAB_EXPERIMENT_ID={scope.experiment_id}",
                              "--env", f"OVLAB_EXPERIMENT_NODE={scope.node_path}"]
                             if scope is not None else [])
        reporting = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--user", "10001:10001", *self._container_groups(self.runs_root, context.root, exports_root, interactive_root),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_MODEL_DATA_ROOT=/var/lib/ovlab/model-data",
            "--env", "OVLAB_RUNS_ROOT=/var/lib/ovlab/runs",
            "--env", "OVLAB_EXPORTS_ROOT=/var/lib/ovlab/exports",
            "--env", f"OVLAB_EXPORTS_DISPLAY_ROOT={exports_root}",
            *scope_environment,
            "--mount", f"type=bind,source={self.runs_root},target=/var/lib/ovlab/runs,readonly",
            "--mount", f"type=bind,source={exports_root},target=/var/lib/ovlab/exports",
            reporting_image, "train", "report", "--run", context.run_id, "--json",
        ]
        legacy_training_runs = self.model_data_root / "training-runs"
        if legacy_training_runs.is_dir():
            image_position = reporting.index(reporting_image)
            reporting[image_position:image_position] = [
                "--mount",
                f"type=bind,source={legacy_training_runs},target=/var/lib/ovlab/model-data/training-runs,readonly",
            ]
        published = self.runner(reporting, capture_output=True, text=True, check=False)
        if published.returncode != 0:
            raise TrainingRuntimeError(
                "canonical training run and checkpoint completed, but isolated performance reporting "
                f"failed with exit code {published.returncode}; evidence: {context.root}"
            )
        inspected = store.inspect(context.run_id)
        try:
            report_result = json.loads(published.stdout).get("result")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise TrainingRuntimeError("training report returned an invalid JSON result") from exc
        return {**inspected, "performance_report": report_result, **({"wandb": tracking} if tracking else {})}
