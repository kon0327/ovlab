"""Immutable file manifest for published full-weight OpenVLA artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CheckpointFileIdentity:
    name: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.name)
        if path.is_absolute() or len(path.parts) != 1 or path.name != self.name:
            raise ValueError("checkpoint file name must be one safe basename")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("checkpoint file size must be positive")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("checkpoint file sha256 must be a lowercase SHA-256 digest")

    def manifest_line(self) -> str:
        return f"{self.name} {self.size} {self.sha256}\n"

    def as_metadata(self) -> dict[str, object]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class OpenVlaRuntimeArtifact:
    logical_resource_id: str
    repository: str
    revision: str
    artifact_form: str
    merge_status: str
    adapter_config_status: str
    adapter_recoverability: str
    files: tuple[CheckpointFileIdentity, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "logical_resource_id", "repository", "revision", "artifact_form", "merge_status",
            "adapter_config_status", "adapter_recoverability",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.artifact_form != "merged_full_weights" or self.merge_status != "merged":
            raise ValueError("Gate D runtime artifact must be merged full weights")
        if self.adapter_config_status != "not_present_in_published_artifact":
            raise ValueError("official merged artifact must report its absent adapter_config.json")
        if self.adapter_recoverability != "not_recoverable_from_published_artifact":
            raise ValueError("official merged artifact recoverability is misclassified")
        files = tuple(self.files)
        if not files or len({item.name for item in files}) != len(files):
            raise ValueError("runtime artifact files must be non-empty and uniquely named")
        required = {"config.json", "model.safetensors.index.json"}
        names = {item.name for item in files}
        if not required <= names or not any(name.endswith(".safetensors") for name in names):
            raise ValueError("runtime artifact manifest lacks config, index, or weight shards")
        if _SHA256.fullmatch(self.aggregate_sha256) is None:
            raise ValueError("aggregate_sha256 must be a lowercase SHA-256 digest")
        calculated = hashlib.sha256("".join(item.manifest_line() for item in files).encode()).hexdigest()
        if calculated != self.aggregate_sha256:
            raise ValueError("aggregate_sha256 does not match the canonical file manifest")
        object.__setattr__(self, "files", files)

    @classmethod
    def from_registry_entry(cls, resource_id: str, entry: dict[str, object]) -> "OpenVlaRuntimeArtifact":
        artifact = entry["artifact"]
        return cls(
            logical_resource_id=resource_id,
            repository=entry["repo_id"],
            revision=entry["expected_revision"],
            artifact_form=artifact["form"],
            merge_status=artifact["merge_status"],
            adapter_config_status=artifact["adapter_config"],
            adapter_recoverability=artifact["adapter_recoverability"],
            files=tuple(
                CheckpointFileIdentity(name=name, **identity)
                for name, identity in artifact["files"].items()
            ),
            aggregate_sha256=entry["expected_sha256"],
        )

    def verify(self, snapshot: Path) -> dict[str, object]:
        root = snapshot.resolve()
        if (root / "adapter_config.json").exists():
            raise ValueError("official merged artifact unexpectedly contains adapter_config.json")
        observed = []
        for expected in self.files:
            path = root / expected.name
            if path.parent != root or not path.is_file():
                raise ValueError(f"checkpoint manifest file is missing: {expected.name}")
            size = path.stat().st_size
            if size != expected.size:
                raise ValueError(
                    f"checkpoint file size mismatch for {expected.name}: expected {expected.size}, got {size}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected.sha256:
                raise ValueError(
                    f"checkpoint SHA-256 mismatch for {expected.name}: expected {expected.sha256}, got {actual}"
                )
            observed.append(expected.as_metadata())
        return {
            "logical_resource_id": self.logical_resource_id,
            "repository": self.repository,
            "revision": self.revision,
            "resolved_snapshot_path": str(root),
            "artifact_form": self.artifact_form,
            "merge_status": self.merge_status,
            "adapter_config": self.adapter_config_status,
            "adapter_recoverability": self.adapter_recoverability,
            "files": observed,
            "aggregate_sha256": self.aggregate_sha256,
        }

    def as_metadata(self) -> dict[str, object]:
        return {
            "logical_resource_id": self.logical_resource_id,
            "repository": self.repository,
            "revision": self.revision,
            "artifact_form": self.artifact_form,
            "merge_status": self.merge_status,
            "adapter_config": self.adapter_config_status,
            "adapter_recoverability": self.adapter_recoverability,
            "files": [item.as_metadata() for item in self.files],
            "aggregate_sha256": self.aggregate_sha256,
        }
