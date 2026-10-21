"""Host owner for isolated dataset acquisition, training, and finalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

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

    def _profile(self, reference: str | Path) -> tuple[Path, TrainingProfile]:
        from .config_paths import resolve_config_path

        path = resolve_config_path(reference, self.repository_root, "experiments/training")
        if not path.is_file():
            raise TrainingRuntimeError(f"training profile does not exist: {path.resolve()}")
        from .training_ablations import load_training_profile
        return path.resolve(), load_training_profile(path.resolve(), config_root=self.repository_root / "configs")

    def _gpu(self) -> tuple[int, float]:
        completed = self.runner(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            return 0, 0.0
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return len(values), max((float(value) / 1024.0 for value in values), default=0.0)

    def _docker_image(self, variable: str, default: str, expected_contract: str) -> tuple[str, str]:
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
        return image, digest

    @staticmethod
    def _container_groups(*directories: Path) -> list[str]:
        """Grant the worker the actual groups of its managed bind mounts."""
        groups = {os.getgid(), *(path.stat().st_gid for path in directories)}
        return [argument for group in sorted(groups) for argument in ("--group-add", str(group))]

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

    def run(self, profile_reference: str | Path, *, allow_dataset_download: bool = False,
            fit_action_normalization: bool = False, interactive_progress: bool = False) -> dict[str, object]:
        for directory in (
            self.model_data_root / "datasets",
            self.runs_root / "train",
            self.model_data_root / "checkpoints",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o2770)
        profile_path, profile = self._profile(profile_reference)
        from .training_profiles import with_fitted_action_normalization
        profile = with_fitted_action_normalization(profile, fit_action_normalization)
        is_quic = profile.document["training"].get("peft", {}).get("method") == "quic"
        training_image, training_digest = self._docker_image(
            "OVLAB_TRAINING_QUIC_IMAGE" if is_quic else "OVLAB_TRAINING_IMAGE",
            "ovlab-training-openvla-quic:local" if is_quic else "ovlab-training-openvla:local",
            "offline-dataset-ro-base-ro-training-run-rw-v1",
        )
        dataset_image, dataset_digest = self._docker_image(
            "OVLAB_DATASET_IMAGE", "ovlab-dataset:local", "dataset-write-only-or-finalizer-v1"
        )
        reporting_image, reporting_digest = self._docker_image(
            "OVLAB_REPORTING_IMAGE", "ovlab-reporting:local", "canonical-readonly-reporting-v1"
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
                self._explicit_dataset_acquisition(profile, dataset_image)
        gpu_count, vram_gib = self._gpu()
        plan = TrainingPlanner(self.repository_root, self.model_data_root).plan(
            profile,
            available_gpu_count=gpu_count,
            available_vram_gib=vram_gib,
            image_identity=training_digest,
            allow_checkpoint_download=True,
        )
        plan = self._prepare_base_checkpoint(plan)
        store = TrainingRunStore(self.model_data_root, runs_root=self.runs_root)
        context = store.create(
            load(profile_path),
            profile.document,
            plan,
            {
                "repository_revision": repository_revision(self.repository_root) or "unavailable",
                "source_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=self.repository_root, capture_output=True, text=True).stdout),
                "training_image": {"reference": training_image, "digest": training_digest},
                "finalizer_image": {"reference": dataset_image, "digest": dataset_digest},
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
        tfds_name = dataset_manifest.get("source_metadata", {}).get("tfds_name") if isinstance(dataset_manifest.get("source_metadata"), dict) else None
        if not isinstance(tfds_name, str):
            raise TrainingRuntimeError("prepared dataset manifest lacks its OpenVLA TFDS name")
        base_path = Path(str(execution["model_host_path"]))
        prepared = Path(str(dataset_manifest["host_path"])) / "prepared"
        container_name = f"ovlab-training-{uuid.uuid4().hex[:12]}"
        # Workers derive the canonical run ID from the mounted root's basename.
        container_run_root = f"/var/lib/ovlab/training-run/{context.run_id}"
        command = [
            "docker", "run", *(["--tty"] if interactive_progress else []),
            "--name", container_name, "--rm", "--network", "none", "--read-only",
            "--gpus", f"device={self.environment.get('OVLAB_GPU_DEVICE', '0')}",
            "--user", "10001:10001", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            *self._container_groups(context.root),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_TRAINING_RUNTIME=isolated-container", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "WANDB_MODE=offline",
            "--env", f"OVLAB_PROGRESS={self.environment.get('OVLAB_PROGRESS', 'auto')}",
            *(["--env", f"OVLAB_TERMINAL_COLUMNS={shutil.get_terminal_size((80, 24)).columns or 80}"] if interactive_progress else []),
            "--env", "OVLAB_TRAINING_BASE_CHECKPOINT=/checkpoints/base",
            "--env", "OVLAB_TRAINING_DATA_ROOT=/datasets/resolved",
            "--env", f"OVLAB_TRAINING_DATASET_NAME={tfds_name}",
            "--mount", f"type=bind,source={base_path},target=/checkpoints/base,readonly",
            "--mount", f"type=bind,source={prepared},target=/datasets/resolved/{tfds_name}/1.0.0,readonly",
            "--mount", f"type=bind,source={context.root},target={container_run_root}",
            "--entrypoint", "python", training_image,
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
            error = TrainingRuntimeError(f"isolated OpenVLA trainer failed with exit code {completed.returncode}; evidence: {context.root}")
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
            "--entrypoint", "python", dataset_image,
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
        derived_root = exports_root / "derived"
        interactive_root = exports_root / "interactive"
        for output_root in (derived_root, interactive_root):
            output_root.mkdir(parents=True, exist_ok=True)
            output_root.chmod(0o2770)
        reporting = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--user", "10001:10001", *self._container_groups(self.runs_root, context.root, exports_root, derived_root, interactive_root),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_MODEL_DATA_ROOT=/var/lib/ovlab/model-data",
            "--env", f"OVLAB_DERIVED_DISPLAY_ROOT={derived_root}",
            "--env", "OVLAB_RUNS_ROOT=/var/lib/ovlab/runs",
            "--env", "OVLAB_EXPORTS_ROOT=/var/lib/ovlab/exports",
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
