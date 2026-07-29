"""Immutable identity and verification for the compound OFT checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class OftFileIdentity:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.path:
            raise ValueError("OFT artifact path must be a safe relative POSIX path")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("OFT artifact size must be positive")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("OFT artifact SHA-256 is invalid")

    def manifest_line(self) -> str:
        return f"{self.path} {self.size} {self.sha256}\n"

    def as_metadata(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class OpenVlaOftArtifact:
    logical_resource_id: str
    repository: str
    revision: str
    aggregate_sha256: str
    files: tuple[OftFileIdentity, ...]
    method: dict[str, object]
    parameter_counts: dict[str, int]
    byte_counts: dict[str, int]

    def __post_init__(self) -> None:
        if self.repository != "moojink/openvla-7b-oft-finetuned-libero-10":
            raise ValueError("Gate E requires the official OpenVLA-OFT LIBERO-10 repository")
        if not isinstance(self.revision, str) or len(self.revision) != 40:
            raise ValueError("Gate E requires a full immutable revision")
        files = tuple(sorted(self.files, key=lambda item: item.path))
        names = {item.path for item in files}
        required = {
            "config.json", "dataset_statistics.json", "model.safetensors.index.json",
            "action_head--150000_checkpoint.pt", "proprio_projector--150000_checkpoint.pt",
            "lora_adapter/adapter_config.json", "lora_adapter/adapter_model.safetensors",
        }
        if not required <= names or len([name for name in names if name.startswith("model-")]) != 4:
            raise ValueError("OFT manifest lacks a required backbone, adapter, head, or projector component")
        calculated = hashlib.sha256("".join(item.manifest_line() for item in files).encode()).hexdigest()
        if calculated != self.aggregate_sha256:
            raise ValueError("OFT aggregate SHA-256 does not match its canonical manifest")
        validate_oft_method(self.method)
        if any(type(value) is not int or value <= 0 for value in self.parameter_counts.values()):
            raise ValueError("OFT parameter counts must be positive integers")
        if any(type(value) is not int or value <= 0 for value in self.byte_counts.values()):
            raise ValueError("OFT byte counts must be positive integers")
        object.__setattr__(self, "files", files)

    @classmethod
    def from_registry_entry(cls, resource_id: str, entry: dict[str, object]) -> "OpenVlaOftArtifact":
        return cls(
            resource_id,
            entry["repo_id"],
            entry["revision"],
            entry["expected_sha256"],
            tuple(
                OftFileIdentity(path, identity["size"], identity["sha256"])
                for path, identity in entry["artifact"]["files"].items()
            ),
            dict(entry["method"]),
            dict(entry["artifact"]["parameter_counts"]),
            dict(entry["artifact"]["byte_counts"]),
        )

    def verify(self, snapshot: Path) -> dict[str, object]:
        root = snapshot.resolve()
        observed = []
        for expected in self.files:
            # Hugging Face snapshots intentionally contain symlinks to cache blobs;
            # validate the lexical relative path, then hash the linked immutable file.
            path = root / expected.path
            if not path.is_file():
                raise ValueError(f"OFT checkpoint file is missing: {expected.path}")
            if path.stat().st_size != expected.size:
                raise ValueError(f"OFT checkpoint size mismatch: {expected.path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected.sha256:
                raise ValueError(f"OFT checkpoint SHA-256 mismatch: {expected.path}")
            observed.append(expected.as_metadata())
        return {**self.as_metadata(), "resolved_snapshot_path": str(root), "files": observed}

    def as_metadata(self) -> dict[str, object]:
        return {
            "logical_resource_id": self.logical_resource_id,
            "repository": self.repository,
            "revision": self.revision,
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.as_metadata() for item in self.files],
            "parameter_counts": dict(self.parameter_counts),
            "byte_counts": dict(self.byte_counts),
        }


def validate_oft_method(method: dict[str, object]) -> None:
    expected = {
        "family": "openvla_oft", "acronym_expansion": "optimized_fine_tuning",
        "backbone_adaptation": "lora", "artifact_form": "merged_backbone_with_auxiliary_components",
        "backbone_merge_status": "merged", "runtime_active_adapter": False,
        "parallel_decoding": True, "action_representation": "continuous",
        "objective": "l1_regression", "action_chunk_size": 8, "action_dimension": 7,
        "normalization": "bounds_q99", "image_inputs": 2, "proprioception_dimension": 8,
        "film": False, "diffusion": False, "quantization": "none",
        "adaptation_suite": "LIBERO-10", "dataset_identity": "libero_10_no_noops",
    }
    mismatches = {key: (method.get(key), value) for key, value in expected.items() if method.get(key) != value}
    if mismatches:
        raise ValueError(f"invalid OpenVLA-OFT methodological classification: {mismatches}")
    forbidden = {"plain_lora", "oft_plus", "quic"} & set(method)
    forbidden |= {key for key in method if key.lower().startswith("qp")}
    if forbidden:
        raise ValueError(f"OFT method contains forbidden classifications: {sorted(forbidden)}")
