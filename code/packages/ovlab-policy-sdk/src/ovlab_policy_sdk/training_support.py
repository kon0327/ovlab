"""Public generic trainer lifecycle and artifact inspection helpers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import struct
from typing import Mapping
from .training_errors import CheckpointBundleError
from .training_identity import canonical_json

TRAINING_RUN_SCHEMA = "ovlab.training-run/v1"
TRAINING_RESULT_SCHEMA = "ovlab.training-result/v1"
CHECKPOINT_SCHEMA = "ovlab.checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA = "ovlab.checkpoint-manifest/v1"
TRAINER_ADAPTER_VERSION = "1.0.0"

def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def append_jsonl(path: Path, document: Mapping):
    with path.open("ab") as stream:
        stream.write(canonical_json(document))
        stream.flush()
        os.fsync(stream.fileno())

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



# Public names; underscored aliases retained only for compatibility.
safetensors_header = _safetensors_header
