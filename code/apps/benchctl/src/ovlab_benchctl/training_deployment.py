"""Host owner for isolated dataset acquisition, training, and finalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import uuid

from .application import OvlabApplication
from .datasets import DatasetRequest, DatasetStore, LiberoDatasetBridge
from .strict_yaml import load
from .training_errors import DatasetUnavailableError, TrainingRuntimeError
from .training_profiles import TrainingProfile, TrainingPlanner
from .training_runs import TrainingRunStore
from .versioning import repository_revision


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

    def _profile(self, reference: str | Path) -> tuple[Path, TrainingProfile]:
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise TrainingRuntimeError(f"training profile does not exist: {path.resolve()}")
        return path.resolve(), TrainingProfile.from_document(load(path.resolve()))

    def _gpu(self) -> tuple[int, float]:
        completed = self.runner(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            return 0, 0.0
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return len(values), max((float(value) / 1024.0 for value in values), default=0.0)

    def _docker_image(self, variable: str, default: str, expected_role: str) -> tuple[str, str]:
        image = self.environment.get(variable, default)
        completed = self.runner(
            ["docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}|{{.Id}}"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise TrainingRuntimeError(f"required image is unavailable: {image}")
        labels_payload, digest = completed.stdout.strip().split("|", 1)
        labels = json.loads(labels_payload)
        if labels.get("cz.cvut.ovlab.role") != expected_role:
            raise TrainingRuntimeError(f"image {image} has role {labels.get('cz.cvut.ovlab.role')!r}, expected {expected_role!r}")
        return image, digest

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
            "--user", "10001:10001", "--group-add", str(os.getgid()),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_MODEL_DATA_ROOT=/var/lib/ovlab/model-data",
            "--mount", f"type=bind,source={datasets_root},target=/var/lib/ovlab/model-data/datasets",
            dataset_image, "dataset", "fetch", "--source", "libero", "--name", name, "--json",
        ]
        completed = self.runner(command, check=False)
        if completed.returncode != 0:
            raise DatasetUnavailableError(f"explicit LIBERO acquisition container failed with exit code {completed.returncode}")

    def run(self, profile_reference: str | Path, *, allow_dataset_download: bool = False) -> dict[str, object]:
        for directory in (
            self.model_data_root / "datasets",
            self.model_data_root / "training-runs",
            self.model_data_root / "checkpoints",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o2770)
        profile_path, profile = self._profile(profile_reference)
        training_image, training_digest = self._docker_image(
            "OVLAB_TRAINING_IMAGE", "ovlab-training-openvla:local", "training-openvla"
        )
        dataset_image, dataset_digest = self._docker_image(
            "OVLAB_DATASET_IMAGE", "ovlab-dataset:local", "dataset"
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
        )
        store = TrainingRunStore(self.model_data_root)
        context = store.create(
            profile.document,
            profile.document,
            plan,
            {
                "repository_revision": repository_revision(self.repository_root) or "unavailable",
                "source_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=self.repository_root, capture_output=True, text=True).stdout),
                "training_image": {"reference": training_image, "digest": training_digest},
                "finalizer_image": {"reference": dataset_image, "digest": dataset_digest},
                "gpu_count": gpu_count,
                "detected_vram_gib": vram_gib,
                "network": "disabled",
                "profile_source": str(profile_path),
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
        command = [
            "docker", "run", "--name", container_name, "--rm", "--network", "none", "--read-only",
            "--gpus", f"device={self.environment.get('OVLAB_GPU_DEVICE', '0')}",
            "--user", "10001:10001", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--group-add", str(os.getgid()),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g,uid=10001,gid=10001,mode=0700",
            "--env", "OVLAB_TRAINING_RUNTIME=isolated-container", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "WANDB_MODE=offline",
            "--env", "OVLAB_TRAINING_BASE_CHECKPOINT=/checkpoints/base",
            "--env", "OVLAB_TRAINING_DATA_ROOT=/datasets/resolved",
            "--env", f"OVLAB_TRAINING_DATASET_NAME={tfds_name}",
            "--mount", f"type=bind,source={base_path},target=/checkpoints/base,readonly",
            "--mount", f"type=bind,source={prepared},target=/datasets/resolved/{tfds_name}/1.0.0,readonly",
            "--mount", f"type=bind,source={context.root},target=/var/lib/ovlab/training-run",
            "--entrypoint", "python", training_image,
            "-m", "ovlab_benchctl.training_worker", "train", "--run-root", "/var/lib/ovlab/training-run",
        ]
        try:
            completed = self.runner(command, check=False)
        except KeyboardInterrupt as exc:
            self.runner(["docker", "stop", "--time", "20", container_name], check=False)
            self.runner(["docker", "rm", "--force", container_name], check=False)
            store.fail(context, exc, interrupted=True)
            raise
        if completed.returncode != 0:
            raise TrainingRuntimeError(f"isolated OpenVLA trainer failed with exit code {completed.returncode}; evidence: {context.root}")
        finalizer = [
            "docker", "run", "--rm", "--network", "none", "--read-only", "--user", "10001:10001",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--group-add", str(os.getgid()),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=0700",
            "--mount", f"type=bind,source={context.root},target=/var/lib/ovlab/training-run",
            "--mount", f"type=bind,source={self.model_data_root / 'checkpoints'},target=/var/lib/ovlab/model-data/checkpoints",
            "--entrypoint", "python", dataset_image,
            "-m", "ovlab_benchctl.training_worker", "finalize", "--run-root", "/var/lib/ovlab/training-run",
            "--model-data-root", "/var/lib/ovlab/model-data",
        ]
        finalized = self.runner(finalizer, check=False)
        if finalized.returncode != 0:
            raise TrainingRuntimeError(f"checkpoint finalizer failed; staged training evidence remains at {context.root}")
        return store.inspect(context.run_id)
