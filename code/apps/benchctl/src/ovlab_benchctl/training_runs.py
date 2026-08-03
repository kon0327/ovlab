"""Trainer contract, canonical training runs, and immutable checkpoint bundles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Mapping
import uuid

from .strict_yaml import dumps
from .training_errors import CheckpointBundleError, TrainingRuntimeError
from .training_identity import atomic_json, canonical_json, identity, inventory, safe_relative, sha256_file


TRAINING_RUN_SCHEMA = "ovlab.training-run/v1"
TRAINING_RESULT_SCHEMA = "ovlab.training-result/v1"
CHECKPOINT_SCHEMA = "ovlab.checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA = "ovlab.checkpoint-manifest/v1"
TRAINER_ADAPTER_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(profile_id: str, scientific_id: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", profile_id).strip("._-") or "training"
    return f"{slug}_{stamp}_{scientific_id.rsplit('-', 1)[-1][:8]}"


@dataclass(slots=True)
class TrainingRunContext:
    run_id: str
    root: Path
    plan: Mapping[str, object]
    close_count: int = 0

    @property
    def staging_checkpoint(self) -> Path:
        return self.root / "staging-checkpoints" / "final"


class TrainerAdapter(ABC):
    """Generic owner-controlled training boundary."""

    @abstractmethod
    def capabilities(self) -> Mapping[str, object]: ...

    @abstractmethod
    def validate_profile(self, profile: Mapping[str, object]) -> None: ...

    @abstractmethod
    def resolve_plan(self, profile: Mapping[str, object], model: Mapping[str, object], dataset: Mapping[str, object]) -> Mapping[str, object]: ...

    @abstractmethod
    def preflight(self, plan: Mapping[str, object]) -> Mapping[str, object]: ...

    @abstractmethod
    def initialize(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> None: ...

    @abstractmethod
    def train(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> Mapping[str, object]: ...

    @abstractmethod
    def finalize(self, run_context: TrainingRunContext) -> Mapping[str, object]: ...

    @abstractmethod
    def interrupt(self, run_context: TrainingRunContext) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class OpenVlaTrainerAdapter(TrainerAdapter):
    """Production OpenVLA reference adapter; heavy imports occur only in train()."""

    def __init__(self) -> None:
        self._context: TrainingRunContext | None = None
        self._closed = False

    def capabilities(self) -> Mapping[str, object]:
        return {
            "id": "openvla-reference",
            "version": TRAINER_ADAPTER_VERSION,
            "modes": ("full", "peft"),
            "peft_methods": ("lora",),
            "lora_targets": ("all-linear",),
            "quantization": ("none",),
            "precision": ("bf16", "fp32"),
            "output_kinds": ("full", "adapter"),
            "network": "disabled",
        }

    def validate_profile(self, profile: Mapping[str, object]) -> None:
        training = profile.get("training")
        if not isinstance(training, dict) or training.get("mode") not in {"full", "peft"}:
            raise TrainingRuntimeError("OpenVLA trainer received an unresolved training mode")

    def resolve_plan(self, profile: Mapping[str, object], model: Mapping[str, object], dataset: Mapping[str, object]) -> Mapping[str, object]:
        return {"profile": dict(profile), "model": dict(model), "dataset": dict(dataset)}

    def preflight(self, plan: Mapping[str, object]) -> Mapping[str, object]:
        capabilities = plan.get("capabilities")
        if not isinstance(capabilities, dict) or capabilities.get("compatible") is not True:
            raise TrainingRuntimeError("training plan is not capability-compatible")
        return {"compatible": True, "issues": []}

    def initialize(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> None:
        if self._context is not None:
            raise TrainingRuntimeError("OpenVLA trainer may be initialized exactly once")
        if os.environ.get("OVLAB_TRAINING_RUNTIME") != "isolated-container":
            raise TrainingRuntimeError("OpenVLA training may run only in the isolated production training container")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise TrainingRuntimeError("OpenVLA training requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        self._context = run_context

    def train(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> Mapping[str, object]:
        if self._context is not run_context or self._closed:
            raise TrainingRuntimeError("OpenVLA trainer is not initialized")
        return _execute_openvla_training(plan, run_context)

    def finalize(self, run_context: TrainingRunContext) -> Mapping[str, object]:
        if not run_context.staging_checkpoint.is_dir():
            raise TrainingRuntimeError("trainer produced no final staging checkpoint")
        return {"staging_checkpoint": str(run_context.staging_checkpoint), "status": "checkpointing"}

    def interrupt(self, run_context: TrainingRunContext) -> None:
        atomic_json(run_context.root / "result.json", {
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": run_context.run_id,
            "status": "interrupted",
            "failure": {"type": "KeyboardInterrupt", "message": "training interrupted"},
        }, exclusive=False, mode=0o644)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._context is not None:
            self._context.close_count += 1
        self._context = None


class TrainingRunStore:
    def __init__(self, model_data_root: str | Path):
        self.model_data_root = Path(model_data_root).expanduser().resolve()
        self.root = self.model_data_root / "training-runs"

    def create(
        self,
        profile: Mapping[str, object],
        resolved_profile: Mapping[str, object],
        plan: Mapping[str, object],
        provenance: Mapping[str, object],
    ) -> TrainingRunContext:
        scientific_id = str(plan["scientific_training_id"])
        run_id = _run_id(str(profile["id"]), scientific_id)
        root = self.root / run_id
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise TrainingRuntimeError(f"completed or staged training run already exists: {run_id}") from exc
        root.chmod(0o770)
        for directory in ("logs", "staging-checkpoints"):
            (root / directory).mkdir(mode=0o770)
        (root / "original-profile.yaml").write_text(dumps(dict(profile)), encoding="utf-8")
        (root / "resolved-profile.yaml").write_text(dumps(dict(resolved_profile)), encoding="utf-8")
        (root / "original-profile.yaml").chmod(0o660)
        (root / "resolved-profile.yaml").chmod(0o660)
        atomic_json(root / "training-plan.json", dict(plan), mode=0o644)
        atomic_json(root / "provenance.json", {
            "schema_version": TRAINING_RUN_SCHEMA,
            "run_id": run_id,
            "created_at": _utc_now(),
            **dict(provenance),
        }, mode=0o644)
        (root / "events.jsonl").touch(mode=0o660)
        (root / "metrics.jsonl").touch(mode=0o660)
        self.event(root, "planned", {"scientific_training_id": scientific_id})
        return TrainingRunContext(run_id=run_id, root=root, plan=plan)

    @staticmethod
    def _append(path: Path, document: Mapping[str, object]) -> None:
        payload = canonical_json(document)
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def event(self, root: Path, state: str, details: Mapping[str, object] | None = None) -> None:
        self._append(root / "events.jsonl", {
            "schema_version": "ovlab.training-event/v1",
            "state": state,
            "timestamp": _utc_now(),
            "details": dict(details or {}),
        })

    def fail(self, context: TrainingRunContext, exc: BaseException, *, interrupted=False) -> dict[str, object]:
        status = "interrupted" if interrupted else "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)[:4000], "first_raw_excerpt": str(exc)[:1000]}
        self.event(context.root, status, failure)
        result = {
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": context.run_id,
            "status": status,
            "checkpoint_id": None,
            "failure": failure,
        }
        atomic_json(context.root / "result.json", result, exclusive=False, mode=0o644)
        return result

    def list(self) -> list[dict[str, object]]:
        if not self.root.is_dir():
            return []
        results = []
        for path in sorted(self.root.glob("*/result.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append({**document, "host_path": str(path.parent)})
        return results

    def inspect(self, run_id: str) -> dict[str, object]:
        root = self.root / run_id
        if not root.is_dir() or not root.resolve().is_relative_to(self.root.resolve()):
            raise TrainingRuntimeError(f"training run does not exist: {run_id}")
        result_path = root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {"status": "running"}
        plan = json.loads((root / "training-plan.json").read_text(encoding="utf-8"))
        return {"schema_version": TRAINING_RUN_SCHEMA, "run_id": run_id, "host_path": str(root), "result": result, "plan": plan}

    def verify(self, run_id: str) -> dict[str, object]:
        root = self.root / run_id
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise TrainingRuntimeError(f"training run is not finalized: {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != TRAINING_RUN_SCHEMA or manifest.get("status") != "completed":
            raise TrainingRuntimeError(f"training run is not completed: {run_id}")
        for entry in manifest.get("files", ()):
            path = root / safe_relative(str(entry["path"]))
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                raise TrainingRuntimeError(f"training run integrity failure: {entry['path']}")
        return {"schema_version": "ovlab.training-verification/v1", "run_id": run_id, "status": "verified", "verified_file_count": len(manifest.get("files", ()))}


def _safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise CheckpointBundleError(f"invalid safetensors header: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > 128 * 1024 * 1024:
            raise CheckpointBundleError(f"unsafe safetensors header length: {path.name}")
        payload = stream.read(length)
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointBundleError(f"invalid safetensors JSON header: {path.name}") from exc
    if not isinstance(header, dict):
        raise CheckpointBundleError(f"safetensors header must be a mapping: {path.name}")
    return header, 8 + length


def _finite_tensor_payload(path: Path, dtype: str, start: int, end: int, data_offset: int) -> bool:
    element_size = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}.get(dtype)
    if element_size is None:
        return True
    if start < 0 or end < start or (end - start) % element_size:
        raise CheckpointBundleError(f"malformed tensor offsets in {path.name}")
    with path.open("rb") as stream:
        stream.seek(data_offset + start)
        remaining = end - start
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                return False
            remaining -= len(block)
            if dtype == "F32":
                if any((value & 0x7F800000) == 0x7F800000 for (value,) in struct.iter_unpack("<I", block)):
                    return False
            elif dtype == "F64":
                if any((value & 0x7FF0000000000000) == 0x7FF0000000000000 for (value,) in struct.iter_unpack("<Q", block)):
                    return False
            elif dtype == "F16":
                if any((value & 0x7C00) == 0x7C00 for (value,) in struct.iter_unpack("<H", block)):
                    return False
            elif dtype == "BF16":
                if any((value & 0x7F80) == 0x7F80 for (value,) in struct.iter_unpack("<H", block)):
                    return False
    return True


def inspect_safetensors(path: Path) -> list[dict[str, object]]:
    header, offset = _safetensors_header(path)
    tensors = []
    for name, descriptor in sorted(header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise CheckpointBundleError(f"invalid tensor descriptor {name!r}")
        dtype, shape, offsets = descriptor.get("dtype"), descriptor.get("shape"), descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise CheckpointBundleError(f"incomplete tensor descriptor {name!r}")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise CheckpointBundleError(f"invalid tensor shape {name!r}")
        start, end = offsets
        element_size = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}.get(dtype)
        expected_bytes = None if element_size is None else math.prod(shape) * element_size
        if (
            type(start) is not int or type(end) is not int
            or offset + end > path.stat().st_size
            or (expected_bytes is not None and end - start != expected_bytes)
            or not _finite_tensor_payload(path, dtype, start, end, offset)
        ):
            raise CheckpointBundleError(f"tensor {name!r} is malformed or non-finite")
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "byte_range": [start, end]})
    if not tensors:
        raise CheckpointBundleError("adapter checkpoint contains no tensors")
    return tensors


class CheckpointBundleStore:
    def __init__(self, model_data_root: str | Path):
        self.model_data_root = Path(model_data_root).expanduser().resolve()
        self.root = self.model_data_root / "checkpoints"

    def finalize(self, context: TrainingRunContext) -> dict[str, object]:
        plan = context.plan
        scientific = plan.get("scientific")
        if not isinstance(scientific, dict):
            raise CheckpointBundleError("training plan lacks scientific identity")
        output = scientific.get("output")
        training = scientific.get("training")
        base = scientific.get("base_checkpoint")
        if not isinstance(output, dict) or not isinstance(training, dict) or not isinstance(base, dict):
            raise CheckpointBundleError("training plan lacks checkpoint semantics")
        staging = context.staging_checkpoint
        weights = staging / "weights-or-adapter"
        processor = staging / "processor"
        if not weights.is_dir():
            raise CheckpointBundleError("staged checkpoint lacks weights-or-adapter")
        kind = str(output.get("output_kind"))
        adapter_config: dict[str, object] | None = None
        tensors: list[dict[str, object]] = []
        if kind == "adapter":
            config_path = weights / "adapter_config.json"
            candidates = sorted(weights.glob("*.safetensors"))
            if not config_path.is_file() or len(candidates) != 1:
                raise CheckpointBundleError("LoRA bundle requires adapter_config.json and one safetensors file")
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            if str(adapter_config.get("peft_type", "")).upper() != "LORA":
                raise CheckpointBundleError("checkpoint is not a LoRA PEFT adapter")
            expected = training.get("peft")
            if not isinstance(expected, dict):
                raise CheckpointBundleError("adapter plan lacks PEFT configuration")
            if adapter_config.get("r") != expected.get("rank") or adapter_config.get("lora_alpha") != expected.get("alpha"):
                raise CheckpointBundleError("adapter configuration does not match resolved LoRA plan")
            targets = adapter_config.get("target_modules")
            if targets != ["all-linear"] and targets != "all-linear":
                raise CheckpointBundleError("adapter target modules do not match OpenVLA all-linear plan")
            tensors = inspect_safetensors(candidates[0])
            if any("lora_" not in str(item["name"]).lower() for item in tensors):
                raise CheckpointBundleError("adapter contains tensors outside declared LoRA targets")
        files, content_digest, total_size = inventory(staging)
        staged_result_path = context.root / "result.json"
        staged_result = (
            json.loads(staged_result_path.read_text(encoding="utf-8"))
            if staged_result_path.is_file()
            else {}
        )
        parameter_counts = {
            "total": staged_result.get("total_parameter_count"),
            "trainable": staged_result.get("trainable_parameter_count"),
            "frozen": staged_result.get("frozen_parameter_count"),
            "adapter": staged_result.get("adapter_parameter_count"),
            "trainable_adapter": staged_result.get("trainable_adapter_parameter_count"),
        }
        semantics = {
            "schema_version": CHECKPOINT_SCHEMA,
            "kind": "peft_adapter" if kind == "adapter" else "full_checkpoint",
            "output_format": "safetensors",
            "base_checkpoint": base,
            "adapter": adapter_config,
            "planned_peft": training.get("peft") if kind == "adapter" else None,
            "parameter_counts": parameter_counts,
            "tensor_inventory": tensors,
            "content_digest": content_digest,
            "processor": "inherited" if not processor.is_dir() else "bundled",
            "merge_status": "unmerged" if kind == "adapter" else "not_applicable",
            "expected_loader": "ovlab-openvla-peft" if kind == "adapter" else "ovlab-openvla-vanilla",
        }
        checkpoint_id = identity("checkpoint", semantics, 32)
        final = self.root / checkpoint_id
        if final.exists():
            existing = self.inspect(checkpoint_id)
            self.verify(checkpoint_id)
            return {**existing, "reused": True}
        temporary = self.root / f".{checkpoint_id}.{uuid.uuid4().hex}.staging"
        self.root.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        try:
            for source in sorted(staging.rglob("*")):
                relative = source.relative_to(staging)
                target = temporary / relative
                if source.is_symlink():
                    raise CheckpointBundleError(f"checkpoint staging contains symlink: {relative}")
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, target)
            checkpoint_document = {**semantics, "checkpoint_id": checkpoint_id, "training_run_id": context.run_id}
            atomic_json(temporary / "checkpoint.json", checkpoint_document, mode=0o644)
            manifest_files, _, manifest_size = inventory(temporary, exclude=("manifest.json",))
            manifest = {
                "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "state": "finalized",
                "kind": semantics["kind"],
                "base_checkpoint": base,
                "merge_status": semantics["merge_status"],
                "adapter": semantics["adapter"],
                "planned_peft": semantics["planned_peft"],
                "parameter_counts": semantics["parameter_counts"],
                "expected_loader": semantics["expected_loader"],
                "content_digest": content_digest,
                "files": manifest_files,
                "total_size": manifest_size,
                "published_at": _utc_now(),
            }
            # The manifest is created last in the hidden staging directory;
            # one rename then publishes a complete, independently verifiable
            # bundle with no visible half-finalized checkpoint window.
            atomic_json(temporary / "manifest.json", manifest)
            for path in sorted(temporary.rglob("*"), reverse=True):
                path.chmod(0o444 if path.is_file() else 0o555)
            temporary.chmod(0o555)
            os.replace(temporary, final)
            final.chmod(0o555)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        result = {
            **staged_result,
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": context.run_id,
            "status": "completed",
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(final),
            "failure": None,
        }
        atomic_json(context.root / "result.json", result, exclusive=False, mode=0o444)
        run_files, run_digest, _ = inventory(context.root, exclude=("manifest.json",))
        atomic_json(context.root / "manifest.json", {
            "schema_version": TRAINING_RUN_SCHEMA,
            "run_id": context.run_id,
            "status": "completed",
            "scientific_training_id": plan["scientific_training_id"],
            "execution_plan_id": plan["execution_plan_id"],
            "checkpoint_id": checkpoint_id,
            "files": run_files,
            "content_digest": run_digest,
            "finalized_at": _utc_now(),
        })
        return {**result, "checkpoint": {**semantics, "checkpoint_id": checkpoint_id}, "reused": False}

    def _manifest(self, checkpoint_id: str) -> tuple[Path, dict[str, object]]:
        root = self.root / checkpoint_id
        manifest_path = root / "manifest.json"
        if not root.is_dir() or not root.resolve().is_relative_to(self.root.resolve()) or not manifest_path.is_file():
            raise CheckpointBundleError(f"finalized checkpoint does not exist: {checkpoint_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA or manifest.get("state") != "finalized":
            raise CheckpointBundleError(f"checkpoint is not finalized: {checkpoint_id}")
        return root, manifest

    def list(self) -> list[dict[str, object]]:
        if not self.root.is_dir():
            return []
        results = []
        for path in sorted(self.root.glob("checkpoint-*/manifest.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if document.get("state") == "finalized":
                results.append({**document, "host_path": str(path.parent)})
        return results

    def inspect(self, checkpoint_id: str) -> dict[str, object]:
        root, manifest = self._manifest(checkpoint_id)
        checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        return {"manifest": manifest, "checkpoint": checkpoint, "host_path": str(root)}

    def verify(self, checkpoint_id: str) -> dict[str, object]:
        root, manifest = self._manifest(checkpoint_id)
        for entry in manifest.get("files", ()):
            path = root / safe_relative(str(entry["path"]))
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                raise CheckpointBundleError(f"checkpoint integrity failure: {entry['path']}")
        checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        if checkpoint.get("kind") == "peft_adapter":
            weight_files = sorted((root / "weights-or-adapter").glob("*.safetensors"))
            if len(weight_files) != 1:
                raise CheckpointBundleError("adapter checkpoint has an invalid weight inventory")
            inspect_safetensors(weight_files[0])
        return {
            "schema_version": "ovlab.checkpoint-verification/v1",
            "checkpoint_id": checkpoint_id,
            "status": "verified",
            "verified_file_count": len(manifest.get("files", ())),
            "host_path": str(root),
        }


def _execute_openvla_training(plan: Mapping[str, object], context: TrainingRunContext) -> Mapping[str, object]:
    """Run the pinned OpenVLA AutoClass training path without merging adapters."""
    # Imports remain below the runtime gate so validate/plan/inspect stay dependency-light.
    import gc
    import hashlib
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
    from ovlab_openvla_common import (
        cuda_allocator_snapshot, estimated_training_compute, parameter_inventory,
        performance_sample, reset_cuda_peak,
    )

    if not torch.cuda.is_available():
        raise TrainingRuntimeError("OpenVLA production training requires CUDA")
    scientific = plan["scientific"]
    execution = plan["execution"]
    assert isinstance(scientific, dict) and isinstance(execution, dict)
    profile = scientific["profile"]
    training = scientific["training"]
    assert isinstance(profile, dict) and isinstance(training, dict)
    dataset_profile = profile["dataset"]
    assert isinstance(dataset_profile, dict)
    base_path = Path(os.environ.get("OVLAB_TRAINING_BASE_CHECKPOINT", "/checkpoints/base"))
    data_root = Path(os.environ.get("OVLAB_TRAINING_DATA_ROOT", "/datasets/resolved"))
    dataset_name = os.environ.get("OVLAB_TRAINING_DATASET_NAME")
    if not base_path.is_dir() or not data_root.is_dir() or not dataset_name:
        raise TrainingRuntimeError("training container mounts are incomplete")
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]) % (2**32))
    torch.cuda.manual_seed_all(int(training["seed"]))
    device = torch.device("cuda:0")
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
    dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float32
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
    initial_hashes: dict[str, str] = {}
    if training["mode"] == "peft":
        peft = training["peft"]
        assert isinstance(peft, dict)
        model = get_peft_model(model, LoraConfig(
            r=int(peft["rank"]), lora_alpha=int(peft["alpha"]),
            lora_dropout=float(peft["dropout"]), target_modules="all-linear",
            init_lora_weights="gaussian", bias="none",
        ))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                initial_hashes[name] = hashlib.sha256(parameter.detach().cpu().float().numpy().tobytes()).hexdigest()
    parameter_counts = parameter_inventory(model.named_parameters())
    trainable = parameter_counts["trainable"]
    frozen = parameter_counts["frozen"]
    if trainable <= 0 or (training["mode"] == "peft" and frozen <= 0):
        raise TrainingRuntimeError("resolved trainable/frozen parameter selection is invalid")
    if training["mode"] == "peft" and parameter_counts["trainable_non_adapter"] != 0:
        raise TrainingRuntimeError("LoRA training selected undeclared non-adapter parameters")
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(training["learning_rate"]))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDataset(
        data_root, dataset_name, transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=max(100, int(training["per_device_batch_size"]) * 8),
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = DataLoader(dataset, batch_size=int(training["per_device_batch_size"]), num_workers=0, collate_fn=collator)
    metrics_path = context.root / "metrics.jsonl"
    events_path = context.root / "events.jsonl"
    steps = 0
    accumulation_started = None
    accumulation_tokens = 0
    accumulation_examples = 0
    initial_memory = cuda_allocator_snapshot(torch, device)
    run_peak_allocated = int(initial_memory.get("peak_allocated_bytes", 0))
    run_peak_reserved = int(initial_memory.get("peak_reserved_bytes", 0))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for batch_index, batch in enumerate(loader):
        if accumulation_started is None:
            torch.cuda.synchronize()
            memory_before = cuda_allocator_snapshot(torch, device)
            reset_cuda_peak(torch, device)
            accumulation_started = __import__("time").perf_counter_ns()
        attention_mask = batch["attention_mask"]
        accumulation_tokens += int(attention_mask.sum().item())
        accumulation_examples += int(attention_mask.shape[0])
        with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                labels=batch["labels"].to(device),
            )
            loss = output.loss
        if not torch.isfinite(loss):
            raise TrainingRuntimeError(f"non-finite loss at microbatch {batch_index}")
        (loss / int(training["gradient_accumulation_steps"])).backward()
        if (batch_index + 1) % int(training["gradient_accumulation_steps"]) != 0:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), math.inf)
        if not torch.isfinite(grad_norm):
            raise TrainingRuntimeError(f"non-finite gradient norm at optimizer step {steps}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        torch.cuda.synchronize()
        elapsed_ms = (__import__("time").perf_counter_ns() - accumulation_started) / 1_000_000
        memory_after = cuda_allocator_snapshot(torch, device)
        run_peak_allocated = max(run_peak_allocated, int(memory_after.get("peak_allocated_bytes", 0)))
        run_peak_reserved = max(run_peak_reserved, int(memory_after.get("peak_reserved_bytes", 0)))
        compute = estimated_training_compute(
            parameter_counts["total"], token_count=accumulation_tokens,
            trainable_parameter_count=parameter_counts["trainable"],
        )
        metric = {
            "schema_version": "ovlab.training-metric/v1",
            "global_step": steps,
            "optimizer_step": steps,
            "timestamp_utc": _utc_now(),
            "training_loss": float(loss.detach().cpu()),
            "learning_rate": float(training["learning_rate"]),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "step_duration_ms": elapsed_ms,
            "examples_per_second": accumulation_examples / (elapsed_ms / 1000.0),
            "gpu_memory_allocated_bytes": memory_after.get("allocated_bytes"),
            "gpu_memory_reserved_bytes": memory_after.get("reserved_bytes"),
            "gpu_memory_peak_bytes": memory_after.get("peak_allocated_bytes"),
            "gpu_memory_peak_reserved_bytes": memory_after.get("peak_reserved_bytes"),
            "estimated_gflops": compute["estimated_gflops"],
            "performance": performance_sample(
                phase="training_optimizer_step",
                parameter_counts=parameter_counts,
                memory_before=memory_before,
                memory_after=memory_after,
                compute=compute,
            ),
            "timing_method": "perf_counter_ns with torch.cuda.synchronize before and after",
        }
        TrainingRunStore._append(metrics_path, metric)
        accumulation_started = None
        accumulation_tokens = 0
        accumulation_examples = 0
        if steps >= int(training["max_steps"]):
            break
    if steps != int(training["max_steps"]):
        raise TrainingRuntimeError(f"dataset exhausted after {steps} optimizer steps")
    checkpoint = context.staging_checkpoint
    weights = checkpoint / "weights-or-adapter"
    processor_dir = checkpoint / "processor"
    auxiliary = checkpoint / "auxiliary"
    weights.mkdir(parents=True)
    processor_dir.mkdir()
    auxiliary.mkdir()
    processor.save_pretrained(processor_dir)
    model.save_pretrained(weights, safe_serialization=True)
    save_dataset_statistics(dataset.dataset_statistics, auxiliary)
    changed = 0
    if training["mode"] == "peft":
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                final_hash = hashlib.sha256(parameter.detach().cpu().float().numpy().tobytes()).hexdigest()
                changed += int(initial_hashes.get(name) != final_hash)
        if changed <= 0:
            raise TrainingRuntimeError("LoRA adapter tensors did not change from initialization")
    result = {
        "schema_version": TRAINING_RESULT_SCHEMA,
        "run_id": context.run_id,
        "status": "checkpointing",
        "optimizer_steps": steps,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": frozen,
        "total_parameter_count": parameter_counts["total"],
        "adapter_parameter_count": parameter_counts["adapter"],
        "trainable_adapter_parameter_count": parameter_counts["trainable_adapter"],
        "changed_adapter_tensor_count": changed if training["mode"] == "peft" else None,
        "peak_vram_bytes": run_peak_allocated,
        "peak_reserved_vram_bytes": run_peak_reserved,
        "performance": {
            "schema_version": "ovlab.performance-summary/v1",
            "parameter_counts": parameter_counts,
            "cuda_allocator": {
                "source": "pytorch-cuda-caching-allocator",
                "peak_allocated_bytes": run_peak_allocated,
                "peak_reserved_bytes": run_peak_reserved,
                "qualification": "process allocator peak across model load and recorded optimizer steps; not whole-device NVML usage",
            },
            "estimated_compute_method": "dense-parameter-token-proxy/training-v1",
        },
        "checkpoint_id": None,
        "failure": None,
    }
    atomic_json(context.root / "result.json", result, exclusive=False, mode=0o644)
    TrainingRunStore._append(events_path, {"schema_version": "ovlab.training-event/v1", "state": "checkpointing", "timestamp": _utc_now(), "details": result})
    del optimizer, loader, dataset, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result
