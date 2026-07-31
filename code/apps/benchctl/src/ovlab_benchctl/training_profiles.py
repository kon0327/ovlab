"""Strict versioned training profiles and deterministic resolved plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from .checkpointing import checkpoint_spec_by_id, verify_checkpoint
from .datasets import DatasetRequest, DatasetStore
from .training_errors import (
    DatasetUnavailableError,
    TrainingPlanError,
    TrainingProfileError,
    TrainingResourceError,
)
from .training_identity import canonical_json, identity


TRAINING_PROFILE_SCHEMA = "ovlab.training-profile/v1"
TRAINING_PLAN_SCHEMA = "ovlab.training-plan/v1"
OPENVLA_TRAINER_VERSION = "1.0.0"


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrainingProfileError(f"{label} must be a mapping")
    return dict(value)


def _keys(document: Mapping[str, object], allowed: set[str], required: set[str], label: str) -> None:
    unknown = sorted(set(document) - allowed)
    missing = sorted(required - set(document))
    if unknown:
        raise TrainingProfileError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise TrainingProfileError(f"{label} is missing required fields: {', '.join(missing)}")


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TrainingProfileError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, label: str, minimum: float, maximum: float, *, lower_open=False) -> float:
    if type(value) not in {int, float}:
        raise TrainingProfileError(f"{label} must be numeric")
    result = float(value)
    if (result <= minimum if lower_open else result < minimum) or result > maximum:
        bracket = "(" if lower_open else "["
        raise TrainingProfileError(f"{label} must be in {bracket}{minimum}, {maximum}]")
    return result


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    profile_id: str
    document: Mapping[str, object]

    @classmethod
    def from_document(cls, value: object) -> "TrainingProfile":
        document = _mapping(value, "training profile")
        _keys(
            document,
            {"schema_version", "id", "trainer", "model", "dataset", "training", "checkpointing", "resources"},
            {"schema_version", "id", "trainer", "model", "dataset", "training", "checkpointing", "resources"},
            "training profile",
        )
        if document["schema_version"] != TRAINING_PROFILE_SCHEMA:
            raise TrainingProfileError(f"training profile schema_version must be {TRAINING_PROFILE_SCHEMA}")
        profile_id = document["id"]
        if not isinstance(profile_id, str) or not profile_id or "/" in profile_id or ".." in profile_id:
            raise TrainingProfileError("training profile id must be a portable non-empty identifier")

        trainer = _mapping(document["trainer"], "trainer")
        _keys(trainer, {"id", "implementation"}, {"id", "implementation"}, "trainer")
        if trainer != {"id": "openvla", "implementation": "openvla-reference"}:
            raise TrainingProfileError("Gate I registers only trainer openvla/openvla-reference")

        model = _mapping(document["model"], "model")
        _keys(model, {"base_checkpoint", "revision", "processor"}, {"base_checkpoint", "revision", "processor"}, "model")
        if not isinstance(model["base_checkpoint"], str) or not model["base_checkpoint"]:
            raise TrainingProfileError("model.base_checkpoint must be a logical resource ID")
        if Path(model["base_checkpoint"]).is_absolute():
            raise TrainingProfileError("model.base_checkpoint must not be an absolute path")
        if not isinstance(model["revision"], str) or len(model["revision"]) != 40:
            raise TrainingProfileError("model.revision must be an immutable 40-character revision")
        if model["processor"] != "inherited":
            raise TrainingProfileError("Gate I OpenVLA profiles require processor: inherited")

        dataset = _mapping(document["dataset"], "dataset")
        _keys(dataset, {"ref", "preparation", "split"}, {"ref", "preparation", "split"}, "dataset")
        if not isinstance(dataset["ref"], str) or not dataset["ref"] or Path(dataset["ref"]).is_absolute():
            raise TrainingProfileError("dataset.ref must be a portable logical or immutable dataset ID")
        if dataset["preparation"] != "openvla-rlds" or dataset["split"] != "train":
            raise TrainingProfileError("OpenVLA Gate I requires preparation openvla-rlds and split train")

        training = _mapping(document["training"], "training")
        _keys(
            training,
            {
                "mode", "peft", "seed", "max_steps", "per_device_batch_size",
                "gradient_accumulation_steps", "learning_rate", "precision",
                "gradient_checkpointing", "quantization",
            },
            {
                "mode", "seed", "max_steps", "per_device_batch_size",
                "gradient_accumulation_steps", "learning_rate", "precision",
                "gradient_checkpointing", "quantization",
            },
            "training",
        )
        mode = training["mode"]
        if mode not in {"full", "peft"}:
            if isinstance(mode, str) and (mode.lower() == "quic" or mode.startswith(("QP", "A"))):
                raise TrainingProfileError("QuIC, QP, and placement-ablation settings belong to Gate J")
            raise TrainingProfileError("training.mode must be full or peft")
        _integer(training["seed"], "training.seed", 0, 2**63 - 1)
        _integer(training["max_steps"], "training.max_steps", 1, 10_000_000)
        _integer(training["per_device_batch_size"], "training.per_device_batch_size", 1, 4096)
        _integer(training["gradient_accumulation_steps"], "training.gradient_accumulation_steps", 1, 65536)
        _number(training["learning_rate"], "training.learning_rate", 0.0, 1.0, lower_open=True)
        if training["precision"] not in {"bf16", "fp32"}:
            raise TrainingProfileError("training.precision must be bf16 or fp32")
        if type(training["gradient_checkpointing"]) is not bool:
            raise TrainingProfileError("training.gradient_checkpointing must be boolean")
        if training["quantization"] != "none":
            raise TrainingProfileError("Gate I reference training supports quantization: none only; QLoRA is not LoRA")
        if mode == "full":
            if "peft" in training:
                raise TrainingProfileError("full training must not contain training.peft")
        else:
            if "peft" not in training:
                raise TrainingProfileError("peft training requires training.peft")
            peft = _mapping(training["peft"], "training.peft")
            _keys(peft, {"method", "target_modules", "rank", "alpha", "dropout"}, {"method", "target_modules", "rank", "alpha", "dropout"}, "training.peft")
            if peft["method"] != "lora":
                raise TrainingProfileError("Gate I registers only the lora PEFT method; QuIC belongs to Gate J")
            if peft["target_modules"] != ["all-linear"]:
                raise TrainingProfileError("OpenVLA reference LoRA requires target_modules: [all-linear]")
            rank = _integer(peft["rank"], "training.peft.rank", 1, 256)
            alpha = _integer(peft["alpha"], "training.peft.alpha", 1, 256)
            if alpha != min(rank, 16):
                raise TrainingProfileError("OpenVLA reference LoRA alpha must equal min(rank, 16)")
            _number(peft["dropout"], "training.peft.dropout", 0.0, 1.0)

        checkpointing = _mapping(document["checkpointing"], "checkpointing")
        _keys(checkpointing, {"save_strategy", "save_optimizer_state", "output_kind"}, {"save_strategy", "save_optimizer_state", "output_kind"}, "checkpointing")
        if checkpointing["save_strategy"] != "final" or type(checkpointing["save_optimizer_state"]) is not bool:
            raise TrainingProfileError("Gate I requires final checkpoint strategy and explicit save_optimizer_state")
        expected_kind = "adapter" if mode == "peft" else "full"
        if checkpointing["output_kind"] != expected_kind:
            raise TrainingProfileError(f"{mode} training requires checkpointing.output_kind: {expected_kind}")

        resources = _mapping(document["resources"], "resources")
        _keys(resources, {"accelerator", "gpu_count", "max_vram_gib", "network"}, {"accelerator", "gpu_count", "max_vram_gib", "network"}, "resources")
        if resources["accelerator"] not in {"cuda", "cpu"}:
            raise TrainingProfileError("resources.accelerator must be cuda or cpu")
        _integer(resources["gpu_count"], "resources.gpu_count", 0, 64)
        _number(resources["max_vram_gib"], "resources.max_vram_gib", 0.0, 1024.0)
        if resources["network"] != "disabled":
            raise TrainingProfileError("training resources.network must be disabled")
        if resources["accelerator"] == "cuda" and resources["gpu_count"] < 1:
            raise TrainingProfileError("CUDA training requires at least one GPU")
        if resources["accelerator"] == "cpu" and resources["gpu_count"] != 0:
            raise TrainingProfileError("CPU training requires gpu_count: 0")

        # Round-trip through canonical JSON to detach mutable caller-owned mappings.
        normalized = json.loads(canonical_json(document))
        return cls(profile_id=profile_id, document=normalized)

    def as_dict(self) -> dict[str, object]:
        return dict(self.document)


def _checkpoint_candidates(model_data_root: Path, repo_id: str, resource_id: str, revision: str) -> tuple[tuple[str, Path], ...]:
    global_root = Path(os.environ.get("OVLAB_GLOBAL_HF_CACHE", "~/.cache/huggingface")).expanduser().resolve()
    hub = global_root / "hub" if (global_root / "hub").is_dir() else global_root
    global_snapshot = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    managed = model_data_root / "checkpoints" / "huggingface" / resource_id / revision
    return (("global-huggingface-cache", global_snapshot), ("ovlab-managed-cache", managed))


class TrainingPlanner:
    """Dependency-light planner: no network, model import, or GPU allocation."""

    def __init__(self, repository_root: Path, model_data_root: Path):
        self.repository_root = repository_root.resolve()
        self.model_data_root = model_data_root.expanduser().resolve()
        self.datasets = DatasetStore(self.model_data_root)

    def _model(self, profile: TrainingProfile) -> dict[str, object]:
        model = profile.document["model"]
        assert isinstance(model, dict)
        resource_id = str(model["base_checkpoint"])
        spec = checkpoint_spec_by_id(self.repository_root, resource_id)
        if model["revision"] != spec.revision:
            raise TrainingPlanError(
                f"profile model revision {model['revision']} does not match registry revision {spec.revision}"
            )
        for source_kind, candidate in _checkpoint_candidates(self.model_data_root, spec.repo_id, resource_id, spec.revision):
            if candidate.is_dir():
                count = verify_checkpoint(spec, candidate)
                return {
                    "resource_id": resource_id,
                    "repository": spec.repo_id,
                    "revision": spec.revision,
                    "aggregate_sha256": spec.expected_sha256,
                    "verified_file_count": count,
                    "source_kind": source_kind,
                    "host_path": str(candidate.resolve()),
                }
        raise TrainingPlanError(
            f"base checkpoint {resource_id!r} is unavailable locally; training planning never downloads checkpoints"
        )

    def _dataset(self, profile: TrainingProfile, *, allow_dataset_download: bool) -> dict[str, object]:
        dataset = profile.document["dataset"]
        assert isinstance(dataset, dict)
        reference = str(dataset["ref"])
        if reference.startswith("dataset-"):
            return self.datasets.inspect(reference)
        try:
            provider, name = reference.split("/", 1)
        except ValueError as exc:
            raise TrainingPlanError("dataset.ref must be an immutable dataset ID or provider/name selector") from exc
        resolution = self.datasets.resolve(DatasetRequest(source=provider, name=name))
        ready = self.datasets.find_resolution(resolution.resolution_id)
        if ready is None:
            suffix = "; run an explicit dataset fetch first" if allow_dataset_download else "; pass --allow-dataset-download only to train run or fetch explicitly"
            raise DatasetUnavailableError(f"dataset selector {reference!r} is not locally ready{suffix}")
        self.datasets.verify(str(ready["dataset_id"]))
        return ready

    @staticmethod
    def _estimated_vram(profile: TrainingProfile) -> float:
        training = profile.document["training"]
        assert isinstance(training, dict)
        # Published OpenVLA reference guidance reports >=27 GiB for non-quantized LoRA.
        return 60.0 if training["mode"] == "full" else 27.0

    def plan(
        self,
        profile: TrainingProfile,
        *,
        allow_dataset_download: bool = False,
        available_gpu_count: int | None = None,
        available_vram_gib: float | None = None,
        image_identity: str = "unavailable",
    ) -> dict[str, object]:
        dataset = self._dataset(profile, allow_dataset_download=allow_dataset_download)
        model = self._model(profile)
        training = dict(profile.document["training"])
        resources = dict(profile.document["resources"])
        estimated_vram = self._estimated_vram(profile)
        if resources["accelerator"] == "cuda":
            if available_gpu_count is not None and int(resources["gpu_count"]) > available_gpu_count:
                raise TrainingResourceError(
                    f"plan requests {resources['gpu_count']} GPUs but only {available_gpu_count} are available"
                )
            if estimated_vram > float(resources["max_vram_gib"]):
                raise TrainingResourceError(
                    f"estimated {estimated_vram:.1f} GiB VRAM exceeds profile limit {float(resources['max_vram_gib']):.1f} GiB"
                )
            if available_vram_gib is not None and estimated_vram > available_vram_gib:
                raise TrainingResourceError(
                    f"estimated {estimated_vram:.1f} GiB VRAM exceeds detected device memory {available_vram_gib:.1f} GiB"
                )
        scientific = {
            "schema_version": TRAINING_PLAN_SCHEMA,
            "profile": profile.document,
            "trainer": {"id": "openvla-reference", "version": OPENVLA_TRAINER_VERSION},
            "trainer_contract": {
                "model_class": "OpenVLAForActionPrediction",
                "processor": "PrismaticProcessor inherited from base checkpoint",
                "prompt_template": "PurePromptBuilder/openvla",
                "dataset_mapping": {
                    "image": "observation.image_primary[0] -> processor image transform",
                    "instruction": "task.language_instruction -> lowercase OpenVLA action prompt",
                    "action": "action[0] -> ActionTokenizer",
                },
                "action_semantics": {
                    "dimension": 7,
                    "training_representation": "OpenVLA action tokens",
                    "normalization": "bounds_q99 from prepared RLDS dataset statistics",
                },
                "optimizer": "torch.optim.AdamW",
                "scheduler": "constant learning rate (no scheduler object)",
                "image_augmentation": False,
            },
            "base_checkpoint": {key: model[key] for key in ("resource_id", "repository", "revision", "aggregate_sha256")},
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "raw_content_digest": dataset["raw_content_digest"],
                "prepared_content_digest": dataset["prepared_content_digest"],
                "preparation": dataset["preparation"],
            },
            "training": training,
            "output": profile.document["checkpointing"],
            "parameter_selection": {
                "trainable": "all parameters" if training["mode"] == "full" else "PEFT LoRA all-linear targets",
                "frozen": "none" if training["mode"] == "full" else "base OpenVLA parameters",
                "trainable_parameter_count": None,
                "frozen_parameter_count": None,
                "count_status": "measured during trainer initialization",
            },
        }
        scientific_id = identity("training-plan", scientific, 32)
        execution = {
            "scientific_training_id": scientific_id,
            "resources": resources,
            "estimated_vram_gib": estimated_vram,
            "image_identity": image_identity,
            "model_source_kind": model["source_kind"],
            "model_host_path": model["host_path"],
            "dataset_host_path": dataset["host_path"],
            "network": "disabled",
        }
        return {
            "schema_version": TRAINING_PLAN_SCHEMA,
            "profile_id": profile.profile_id,
            "scientific_training_id": scientific_id,
            "execution_plan_id": identity("training-execution", execution, 32),
            "scientific": scientific,
            "execution": execution,
            "capabilities": {
                "compatible": True,
                "training_mode": training["mode"],
                "peft_method": training.get("peft", {}).get("method") if isinstance(training.get("peft"), dict) else None,
                "quantization": "none",
                "network": "disabled",
            },
            "warnings": [],
        }
