"""Dataset provider contracts without storage, network, or orchestration code.

Providers own source-specific resolution/acquisition/preparation. The caller owns
staging, publication, content inventory and lifecycle policy. Resolution identity
retains the public dataset-resolution/v1 representation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

Metadata = Mapping[str, object]

DATASET_RESOLUTION_SCHEMA = "ovlab.dataset-resolution/v1"
DatasetProgress = Callable[[str], None]
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


from .dataset_errors import (
    DatasetError, DatasetRequestError, DatasetUnavailableError, DatasetIntegrityError,
    DatasetSecurityError, DatasetInterruptedError,
)


def _name(value: object, label: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise DatasetRequestError(f"{label} must be a path-safe identifier")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DatasetRequestError(f"{label} must be a non-empty string")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze_metadata(value):
    """Dataset identities contain JSON data and need no numeric runtime import."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise DatasetRequestError("dataset metadata keys must be strings")
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise DatasetRequestError("dataset metadata must contain finite JSON values")


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    source: str
    name: str
    version: str = "1"
    url: str | None = None
    sha256: str | None = None
    archive: str = "auto"
    preparation: str | None = None
    local_path: Path | None = None
    allow_local_http: bool = False

    def __post_init__(self) -> None:
        for label in ("source", "name", "version"):
            _name(getattr(self, label), f"dataset {label}")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None):
            raise DatasetRequestError("dataset SHA-256 must contain 64 lowercase hexadecimal characters")
        if self.archive not in {"auto", "none", "zip", "tar", "tar.gz", "tgz", "tar.zst"}:
            raise DatasetRequestError("unsupported archive format")
        for label in ("url", "preparation"):
            if getattr(self, label) is not None:
                _text(getattr(self, label), f"dataset {label}")
        if self.local_path is not None and not isinstance(self.local_path, Path):
            raise DatasetRequestError("dataset local_path must be a Path or None")
        if type(self.allow_local_http) is not bool:
            raise DatasetRequestError("allow_local_http must be a boolean")


@dataclass(frozen=True, slots=True)
class DatasetResolution:
    provider: str
    logical_name: str
    source_revision: str
    source_type: str
    canonical_locator: str
    bridge_id: str
    bridge_version: str
    expected_digest: str | None
    preparation_format: str
    source_metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("provider", "logical_name", "bridge_id", "bridge_version"):
            _name(getattr(self, label), label)
        for label in ("source_revision", "source_type", "canonical_locator", "preparation_format"):
            _text(getattr(self, label), label)
        if self.expected_digest is not None and (
            not isinstance(self.expected_digest, str) or _SHA256.fullmatch(self.expected_digest) is None
        ):
            raise DatasetRequestError("expected_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.source_metadata, Mapping):
            raise DatasetRequestError("source_metadata must be a mapping")
        object.__setattr__(self, "source_metadata", _freeze_metadata(self.source_metadata))

    def identifying_document(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_RESOLUTION_SCHEMA,
            "provider": self.provider,
            "logical_name": self.logical_name,
            "source_revision": self.source_revision,
            "source_type": self.source_type,
            "canonical_locator": self.canonical_locator,
            "bridge": {"id": self.bridge_id, "version": self.bridge_version},
            "expected_digest": self.expected_digest,
            "preparation_format": self.preparation_format,
            "source_metadata": _plain(self.source_metadata),
        }

    @property
    def resolution_id(self) -> str:
        canonical = json.dumps(self.identifying_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        return "dsr-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def as_dict(self) -> dict[str, object]:
        return {**self.identifying_document(), "resolution_id": self.resolution_id, "state": "resolved"}


class DatasetProvider(ABC):
    """Source-specific hooks; only explicit acquisition/preparation performs IO.

    ``supports_resume`` tells the generic store whether the provider can verify
    and reuse partially acquired files. It replaces checks for concrete classes.
    Providers must reject files outside their immutable resolution, including
    unsafe paths and unexpected pre-existing staging bytes.
    """

    api_version = "1"
    bridge_id: str
    bridge_version: str
    supports_resume: bool = False

    def publication_version(self, resolution: DatasetResolution) -> str:
        """Path-safe public version; independent of the source revision/pin."""
        return resolution.source_revision

    @abstractmethod
    def capabilities(self) -> Mapping[str, object]: ...

    @abstractmethod
    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        """Describe immutable data without downloading it."""

    @abstractmethod
    def acquire(self, resolution: DatasetResolution, destination: Path,
                progress: DatasetProgress | None = None) -> None: ...

    def verify(self, resolution: DatasetResolution, acquired: Path) -> dict[str, object]:
        from .artifact_identity import inventory
        files, digest, size = inventory(acquired, exclude=("source.json",))
        if not files:
            raise DatasetIntegrityError("acquired dataset contains no files")
        if resolution.expected_digest is not None and digest != resolution.expected_digest:
            raise DatasetIntegrityError(f"dataset content digest mismatch: expected {resolution.expected_digest}, got {digest}")
        return {"files": files, "content_digest": digest, "total_size": size}

    @abstractmethod
    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]: ...
