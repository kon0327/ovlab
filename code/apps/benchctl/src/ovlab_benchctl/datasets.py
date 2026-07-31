"""Versioned dataset registry, bridges, acquisition, and immutable storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import time
from typing import Callable, Iterator, Mapping
from urllib.error import URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit
from urllib.request import Request, build_opener
import uuid
import zipfile

from .training_errors import (
    DatasetIntegrityError,
    DatasetInterruptedError,
    DatasetRequestError,
    DatasetSecurityError,
    DatasetUnavailableError,
)
from .training_identity import (
    SAFE_ID_RE,
    SHA256_RE,
    atomic_json,
    canonical_json,
    identity,
    inventory,
    redact_url,
    safe_relative,
    sha256_file,
)


DATASET_SOURCE_SCHEMA = "ovlab.dataset-source/v1"
DATASET_RESOLUTION_SCHEMA = "ovlab.dataset-resolution/v1"
DATASET_MANIFEST_SCHEMA = "ovlab.dataset-manifest/v1"
DATASET_PREPARATION_SCHEMA = "ovlab.dataset-preparation/v1"
LIBERO_BRIDGE_VERSION = "1.0.0"
URL_BRIDGE_VERSION = "1.0.0"
LOCAL_BRIDGE_VERSION = "1.0.0"

Progress = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise DatasetRequestError(f"{label} must be a path-safe identifier")
    return value


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
        _validate_name(self.source, "dataset source")
        _validate_name(self.name, "dataset name")
        _validate_name(self.version, "dataset version")
        if self.sha256 is not None and SHA256_RE.fullmatch(self.sha256) is None:
            raise DatasetRequestError("dataset SHA-256 must contain 64 lowercase hexadecimal characters")
        if self.archive not in {"auto", "none", "zip", "tar", "tar.gz", "tgz", "tar.zst"}:
            raise DatasetRequestError("archive must be auto, none, zip, tar, tar.gz, tgz, or tar.zst")


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
    source_metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def resolution_id(self) -> str:
        return identity("dsr", self.identifying_document())

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
            "source_metadata": dict(self.source_metadata),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.identifying_document(), "resolution_id": self.resolution_id, "state": "resolved"}


class DatasetBridge(ABC):
    """Training-independent dataset provider boundary."""

    bridge_id: str
    bridge_version: str

    @abstractmethod
    def capabilities(self) -> Mapping[str, object]: ...

    @abstractmethod
    def resolve(self, request: DatasetRequest) -> DatasetResolution: ...

    @abstractmethod
    def acquire(self, resolution: DatasetResolution, destination: Path, progress: Progress | None = None) -> None: ...

    def verify(self, resolution: DatasetResolution, acquired: Path) -> dict[str, object]:
        # Acquisition provenance is deliberately outside the dataset payload
        # identity.  Otherwise adding the source record after downloading would
        # make a provider's expected content digest impossible to satisfy.
        files, digest, size = inventory(acquired, exclude=("source.json",))
        if not files:
            raise DatasetIntegrityError("acquired dataset contains no files")
        if resolution.expected_digest is not None and digest != resolution.expected_digest:
            raise DatasetIntegrityError(
                f"dataset content digest mismatch: expected {resolution.expected_digest}, got {digest}"
            )
        return {"files": files, "content_digest": digest, "total_size": size}

    @abstractmethod
    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]: ...


class _HttpMixin:
    timeout_s = 30.0
    retries = 3

    @staticmethod
    def _validate_url(url: str, *, allow_local_http: bool = False) -> None:
        parsed = urlsplit(url)
        if parsed.username is not None or parsed.password is not None:
            raise DatasetSecurityError("dataset URL must not embed credentials")
        if parsed.scheme != "https":
            local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if not (allow_local_http and local):
                raise DatasetSecurityError("dataset URL must use HTTPS")
        if not parsed.hostname:
            raise DatasetSecurityError("dataset URL requires a hostname")

    def _open(self, url: str, *, allow_local_http: bool = False):
        self._validate_url(url, allow_local_http=allow_local_http)
        response = build_opener().open(Request(url, headers={"User-Agent": "OVLAB-dataset/1"}), timeout=self.timeout_s)
        final = response.geturl()
        self._validate_url(final, allow_local_http=allow_local_http)
        return response

    def _download(
        self,
        url: str,
        target: Path,
        *,
        expected_sha256: str | None,
        allow_local_http: bool = False,
        progress: Progress | None = None,
        maximum_bytes: int = 256 * 1024 * 1024 * 1024,
    ) -> tuple[str, str, int]:
        partial = target.with_name(target.name + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self._open(url, allow_local_http=allow_local_http) as response, partial.open("wb") as stream:
                    final_url = response.geturl()
                    total = 0
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > maximum_bytes:
                            raise DatasetSecurityError("dataset download exceeds configured size limit")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                actual = sha256_file(partial)
                if expected_sha256 is not None and actual != expected_sha256:
                    raise DatasetIntegrityError(
                        f"download checksum mismatch: expected {expected_sha256}, got {actual}"
                    )
                os.replace(partial, target)
                if progress is not None:
                    progress(f"downloaded {target.name} ({total} bytes)")
                return actual, redact_url(final_url), total
            except KeyboardInterrupt as exc:
                raise DatasetInterruptedError("dataset download interrupted; partial bytes remain staged") from exc
            except DatasetIntegrityError:
                partial.unlink(missing_ok=True)
                raise
            except (OSError, URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise DatasetUnavailableError(f"dataset download failed after {self.retries} attempts: {last_error}")


class UrlDatasetBridge(_HttpMixin, DatasetBridge):
    bridge_id = "url"
    bridge_version = URL_BRIDGE_VERSION

    def capabilities(self) -> Mapping[str, object]:
        return {"source": "url", "archives": ("none", "zip", "tar", "tar.gz", "tgz", "tar.zst"), "requires_digest": True}

    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        if request.url is None or request.sha256 is None:
            raise DatasetRequestError("URL datasets require --url and --sha256")
        self._validate_url(request.url, allow_local_http=request.allow_local_http)
        return DatasetResolution(
            provider="url",
            logical_name=request.name,
            source_revision=request.version,
            source_type="https-url" if urlsplit(request.url).scheme == "https" else "local-http-fixture",
            canonical_locator=redact_url(request.url),
            bridge_id=self.bridge_id,
            bridge_version=self.bridge_version,
            expected_digest=request.sha256,
            preparation_format=request.preparation or "directory-v1",
            source_metadata={"archive": request.archive, "allow_local_http": request.allow_local_http},
        )

    def acquire(self, resolution: DatasetResolution, destination: Path, progress: Progress | None = None) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        archive = str(resolution.source_metadata["archive"])
        suffix = Path(urlsplit(resolution.canonical_locator).path).name or "dataset.bin"
        target = destination / suffix
        actual_digest, final_origin, total_size = self._download(
            resolution.canonical_locator,
            target,
            expected_sha256=resolution.expected_digest,
            allow_local_http=bool(resolution.source_metadata.get("allow_local_http", False)),
            progress=progress,
        )
        atomic_json(destination / "source.json", {
            "schema_version": DATASET_SOURCE_SCHEMA,
            "canonical_locator": resolution.canonical_locator,
            "source_revision": resolution.source_revision,
            "sha256": actual_digest,
            "archive": archive,
            "final_origin": final_origin,
            "downloaded_size": total_size,
        }, mode=0o644)

    def verify(self, resolution: DatasetResolution, acquired: Path) -> dict[str, object]:
        payloads = [path for path in acquired.iterdir() if path.is_file() and path.name != "source.json"]
        if len(payloads) != 1:
            raise DatasetIntegrityError("URL acquisition must contain exactly one payload")
        digest = sha256_file(payloads[0])
        if digest != resolution.expected_digest:
            raise DatasetIntegrityError(
                f"download checksum mismatch: expected {resolution.expected_digest}, got {digest}"
            )
        return {
            "files": [{"path": payloads[0].name, "size": payloads[0].stat().st_size, "sha256": digest}],
            "content_digest": digest,
            "total_size": payloads[0].stat().st_size,
        }

    @staticmethod
    def _member_path(name: str, root: Path) -> Path:
        relative = safe_relative(name)
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise DatasetSecurityError(f"archive member escapes extraction root: {name}")
        return candidate

    def _extract(self, payload: Path, archive: str, destination: Path, *, max_files=200_000, max_bytes=512 * 1024**3) -> None:
        if archive == "auto":
            lower = payload.name.lower()
            archive = "zip" if lower.endswith(".zip") else "tar.gz" if lower.endswith((".tar.gz", ".tgz")) else "tar.zst" if lower.endswith(".tar.zst") else "tar" if lower.endswith(".tar") else "none"
        if archive == "none":
            shutil.copy2(payload, destination / payload.name)
            return
        count = total = 0
        if archive == "zip":
            with zipfile.ZipFile(payload) as handle:
                for member in handle.infolist():
                    count += 1
                    total += member.file_size
                    if count > max_files or total > max_bytes:
                        raise DatasetSecurityError("archive exceeds extraction limits")
                    self._member_path(member.filename, destination)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise DatasetSecurityError(f"archive symlink is forbidden: {member.filename}")
                handle.extractall(destination)
            return
        if archive == "tar.zst":
            try:
                import zstandard  # type: ignore
            except ImportError as exc:
                raise DatasetUnavailableError("tar.zst acquisition requires the pinned zstandard runtime") from exc
            unpacked = destination.parent / "payload.tar"
            with payload.open("rb") as source, unpacked.open("wb") as target:
                zstandard.ZstdDecompressor().copy_stream(source, target)
            payload, archive = unpacked, "tar"
        mode = "r:gz" if archive in {"tar.gz", "tgz"} else "r:"
        try:
            with tarfile.open(payload, mode) as handle:
                members = handle.getmembers()
                for member in members:
                    count += 1
                    total += member.size
                    if count > max_files or total > max_bytes:
                        raise DatasetSecurityError("archive exceeds extraction limits")
                    self._member_path(member.name, destination)
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise DatasetSecurityError(f"unsafe archive member type: {member.name}")
                handle.extractall(destination, members=members, filter="data")
        except TypeError:
            # Python 3.10 lacks tarfile's extraction filter; all members were validated above.
            with tarfile.open(payload, mode) as handle:
                handle.extractall(destination, members=handle.getmembers())

    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]:
        destination.mkdir(parents=True, exist_ok=False)
        payloads = [path for path in acquired.iterdir() if path.is_file() and path.name != "source.json"]
        if len(payloads) != 1:
            raise DatasetIntegrityError("URL acquisition payload is ambiguous")
        self._extract(payloads[0], str(resolution.source_metadata["archive"]), destination)
        files, digest, size = inventory(destination)
        if not files:
            raise DatasetIntegrityError("prepared dataset contains no files")
        return {"files": files, "content_digest": digest, "total_size": size, "format": resolution.preparation_format}


class LiberoDatasetBridge(_HttpMixin, DatasetBridge):
    bridge_id = "libero"
    bridge_version = LIBERO_BRIDGE_VERSION
    repository = "openvla/modified_libero_rlds"
    revision = "a7c9ae18499b6eea8a32f78a9302327b752b1b5f"
    suites = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    _shard_name = re.compile(r"^.+\.tfrecord-(\d{5})-of-(\d{5})$")

    def capabilities(self) -> Mapping[str, object]:
        return {"source": "libero", "suites": tuple(self.suites), "preparation_formats": ("openvla-rlds",)}

    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        if request.name not in self.suites:
            raise DatasetRequestError(
                f"unknown LIBERO suite {request.name!r}; supported suites: {', '.join(self.suites)}"
            )
        folder = self.suites[request.name]
        return DatasetResolution(
            provider="libero",
            logical_name=request.name,
            source_revision=self.revision,
            source_type="huggingface-dataset-snapshot",
            canonical_locator=f"https://huggingface.co/datasets/{self.repository}/tree/{self.revision}/{folder}/1.0.0",
            bridge_id=self.bridge_id,
            bridge_version=self.bridge_version,
            expected_digest=None,
            preparation_format="openvla-rlds",
            source_metadata={
                "repository": self.repository,
                "snapshot_revision": self.revision,
                "dataset_directory": f"{folder}/1.0.0",
                "tfds_name": folder,
                "libero_revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
                "license": "MIT",
                "citation": "OpenVLA (arXiv:2406.09246) and LIBERO",
            },
        )

    def _tree(self, resolution: DatasetResolution) -> list[dict[str, object]]:
        prefix = str(resolution.source_metadata["dataset_directory"])
        endpoint = (
            f"https://huggingface.co/api/datasets/{self.repository}/tree/"
            f"{self.revision}/{prefix}?" + urlencode({"recursive": "true", "expand": "true"})
        )
        entries: list[dict[str, object]] = []
        next_url: str | None = endpoint
        while next_url:
            with self._open(next_url) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise DatasetUnavailableError("Hugging Face dataset tree response is not a list")
                entries.extend(item for item in payload if isinstance(item, dict) and item.get("type") == "file")
                link = response.headers.get("Link", "")
                next_url = None
                for part in link.split(","):
                    if 'rel="next"' in part and "<" in part and ">" in part:
                        next_url = part.split("<", 1)[1].split(">", 1)[0]
                        break
        if not entries:
            raise DatasetUnavailableError("pinned LIBERO dataset snapshot contains no files")
        return entries

    def acquire(self, resolution: DatasetResolution, destination: Path, progress: Progress | None = None) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or not destination.is_dir():
            raise DatasetSecurityError("LIBERO staging destination must be a real directory")
        prefix = str(resolution.source_metadata["dataset_directory"]).rstrip("/") + "/"
        expected_paths: set[Path] = set()
        for entry in self._tree(resolution):
            remote_path = str(entry.get("path", ""))
            if not remote_path.startswith(prefix):
                raise DatasetSecurityError(f"provider returned a file outside requested LIBERO suite: {remote_path}")
            relative = safe_relative(remote_path[len(prefix):])
            if relative in expected_paths:
                raise DatasetIntegrityError(f"provider returned duplicate LIBERO path: {relative}")
            expected_paths.add(relative)
            lfs = entry.get("lfs", {}) if isinstance(entry.get("lfs"), dict) else {}
            expected = lfs.get("oid") if isinstance(lfs.get("oid"), str) else None
            if isinstance(expected, str) and expected.startswith("sha256:"):
                expected = expected[7:]
            if expected is not None and SHA256_RE.fullmatch(expected) is None:
                expected = None
            target = destination / relative
            partial = target.with_name(target.name + ".partial")
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise DatasetSecurityError(f"unsafe partial LIBERO staging path: {relative}")
            partial.unlink(missing_ok=True)
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise DatasetSecurityError(f"unsafe existing LIBERO staging path: {relative}")
            expected_size = entry.get("size")
            size_matches = not isinstance(expected_size, int) or (
                target.is_file() and target.stat().st_size == expected_size
            )
            if target.is_file() and expected is not None and size_matches and sha256_file(target) == expected:
                if progress is not None:
                    progress(f"reused {relative.as_posix()} ({target.stat().st_size} bytes)")
                continue
            encoded = "/".join(quote(part, safe="") for part in remote_path.split("/"))
            url = f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{encoded}?download=true"
            self._download(url, target, expected_sha256=expected, progress=progress)
        actual_paths = {
            path.relative_to(destination)
            for path in destination.rglob("*")
            if path.is_file() and path.name != "source.json"
        }
        unexpected = actual_paths - expected_paths
        if unexpected:
            names = ", ".join(sorted(path.as_posix() for path in unexpected))
            raise DatasetIntegrityError(f"LIBERO staging contains files outside the pinned snapshot: {names}")
        atomic_json(destination / "source.json", {
            "schema_version": DATASET_SOURCE_SCHEMA,
            "repository": self.repository,
            "revision": self.revision,
            "dataset_directory": resolution.source_metadata["dataset_directory"],
        }, exclusive=False, mode=0o644)

    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]:
        destination.mkdir(parents=True, exist_ok=False)
        required = {"dataset_info.json", "features.json"}
        names = {path.name for path in acquired.iterdir() if path.is_file()}
        shards = [self._shard_name.fullmatch(name) for name in names if ".tfrecord-" in name]
        if not required <= names or not shards or any(match is None for match in shards):
            raise DatasetIntegrityError("LIBERO OpenVLA RLDS snapshot lacks TFDS metadata or TFRecord shards")
        totals = {int(match.group(2)) for match in shards if match is not None}
        indices = {int(match.group(1)) for match in shards if match is not None}
        if len(totals) != 1 or indices != set(range(next(iter(totals)))):
            raise DatasetIntegrityError("LIBERO OpenVLA RLDS snapshot has an incomplete TFRecord shard set")
        for source in sorted(acquired.iterdir()):
            if source.name == "source.json" or not source.is_file():
                continue
            os.link(source, destination / source.name)
        files, digest, size = inventory(destination)
        info = json.loads((destination / "dataset_info.json").read_text(encoding="utf-8"))
        sample_count = sum(int(value) for split in info.get("splits", ()) for value in split.get("shardLengths", ()))
        return {
            "files": files,
            "content_digest": digest,
            "total_size": size,
            "format": "openvla-rlds",
            "sample_count": sample_count,
            "schema": {
                "observation": "steps.observation.image:uint8[256,256,3]",
                "instruction": "steps.language_instruction:string",
                "action": "steps.action:float32[7]",
            },
        }


class LocalDatasetBridge(DatasetBridge):
    bridge_id = "local"
    bridge_version = LOCAL_BRIDGE_VERSION

    def capabilities(self) -> Mapping[str, object]:
        return {"source": "local", "import_modes": ("copy",), "network": False}

    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        if request.local_path is None:
            raise DatasetRequestError("local import requires --path")
        source = request.local_path.expanduser().resolve()
        if not source.is_dir():
            raise DatasetRequestError(f"local dataset path is not a directory: {source}")
        _, digest, _ = inventory(source)
        return DatasetResolution(
            provider="local",
            logical_name=request.name,
            source_revision=request.version,
            source_type="local-directory-import",
            canonical_locator="local-import://redacted",
            bridge_id=self.bridge_id,
            bridge_version=self.bridge_version,
            expected_digest=digest,
            preparation_format=request.preparation or "directory-v1",
            source_metadata={"import_mode": "copy", "source_path_recorded": False, "runtime_accessible": True},
        )

    def acquire(self, resolution: DatasetResolution, destination: Path, progress: Progress | None = None) -> None:
        raise DatasetRequestError("local bridge acquisition requires import_path()")

    def import_path(self, source: Path, destination: Path) -> None:
        source = source.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_symlink():
                raise DatasetSecurityError(f"local import rejects symlink: {relative}")
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]:
        destination.mkdir(parents=True, exist_ok=False)
        for path in sorted(acquired.rglob("*")):
            relative = path.relative_to(acquired)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
        files, digest, size = inventory(destination)
        return {"files": files, "content_digest": digest, "total_size": size, "format": resolution.preparation_format}


class DatasetBridgeRegistry:
    def __init__(self, bridges: tuple[DatasetBridge, ...] | None = None):
        bridges = bridges or (LiberoDatasetBridge(), UrlDatasetBridge(), LocalDatasetBridge())
        self._bridges = {bridge.bridge_id: bridge for bridge in bridges}

    def providers(self) -> list[dict[str, object]]:
        return [
            {"id": key, "version": bridge.bridge_version, "capabilities": dict(bridge.capabilities())}
            for key, bridge in sorted(self._bridges.items())
        ]

    def get(self, source: str) -> DatasetBridge:
        try:
            return self._bridges[source]
        except KeyError as exc:
            raise DatasetRequestError(
                f"unknown dataset provider {source!r}; supported providers: {', '.join(sorted(self._bridges))}"
            ) from exc


class DatasetStore:
    """Owns immutable dataset publication below one model-data root."""

    def __init__(self, model_data_root: str | Path, registry: DatasetBridgeRegistry | None = None):
        self.model_data_root = Path(model_data_root).expanduser().resolve()
        self.root = self.model_data_root / "datasets"
        self.display_root = Path(os.environ.get("OVLAB_MODEL_DATA_DISPLAY_ROOT", str(self.model_data_root)))
        self.registry = registry or DatasetBridgeRegistry()

    def _display(self, path: Path) -> str:
        try:
            return str(self.display_root / path.resolve().relative_to(self.model_data_root))
        except ValueError:
            return str(path)

    @contextmanager
    def _lock(self, resolution_id: str) -> Iterator[None]:
        lock_root = self.root / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        path = lock_root / f"{resolution_id}.lock"
        with path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        return self.registry.get(request.source).resolve(request)

    @staticmethod
    def _storage_version(resolution: DatasetResolution) -> str:
        """Return the portable dataset version used in the public store path.

        A provider revision identifies source provenance, but is not necessarily
        the dataset's user-facing version.  LIBERO is pinned by a Hugging Face
        commit while its published TFDS dataset version is the final component
        of ``dataset_directory`` (currently ``1.0.0``).
        """
        version = resolution.source_revision
        if resolution.provider == "libero":
            dataset_directory = resolution.source_metadata.get("dataset_directory")
            if not isinstance(dataset_directory, str):
                raise DatasetIntegrityError("LIBERO resolution lacks its versioned dataset directory")
            parts = safe_relative(dataset_directory).parts
            if len(parts) < 2:
                raise DatasetIntegrityError("LIBERO dataset directory lacks a dataset version")
            version = parts[-1]
        return _validate_name(version, "dataset version")

    @staticmethod
    def _manifest_version(path: Path, document: Mapping[str, object]) -> str:
        recorded = document.get("dataset_version")
        if isinstance(recorded, str) and SAFE_ID_RE.fullmatch(recorded) is not None:
            return recorded
        metadata = document.get("source_metadata")
        if document.get("provider") == "libero" and isinstance(metadata, Mapping):
            dataset_directory = metadata.get("dataset_directory")
            if isinstance(dataset_directory, str):
                parts = safe_relative(dataset_directory).parts
                if len(parts) >= 2 and SAFE_ID_RE.fullmatch(parts[-1]) is not None:
                    return parts[-1]
        source_revision = document.get("source_revision")
        if isinstance(source_revision, str) and SAFE_ID_RE.fullmatch(source_revision) is not None:
            return source_revision
        raise DatasetIntegrityError(f"dataset manifest has no path-safe version: {path}")

    def _publication_path(self, resolution: DatasetResolution) -> Path:
        return self.root / resolution.provider / resolution.logical_name / self._storage_version(resolution)

    def _assert_version_slot_available(self, resolution: DatasetResolution) -> None:
        final = self._publication_path(resolution)
        if not final.exists():
            return
        manifest_path = final / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetIntegrityError(
                f"dataset version path already exists without a valid manifest: {final}"
            ) from exc
        if (
            manifest.get("schema_version") == DATASET_MANIFEST_SCHEMA
            and manifest.get("state") == "ready"
            and manifest.get("resolution_id") == resolution.resolution_id
        ):
            raise DatasetIntegrityError(
                f"dataset version {self._storage_version(resolution)!r} is already published but was not reusable; "
                f"verify its manifest at {final}"
            )
        raise DatasetIntegrityError(
            f"dataset version {self._storage_version(resolution)!r} for "
            f"{resolution.provider}/{resolution.logical_name} is already bound to a different immutable identity; "
            "choose a new dataset version instead of overwriting it"
        )

    def _manifests(self) -> Iterator[tuple[Path, dict[str, object]]]:
        if not self.root.is_dir():
            return
        # Three levels are the canonical provider/name/version layout.  The
        # four-level revision/build layout remains readable so existing stores
        # can be migrated without losing identity or requiring a download.
        paths = {
            *self.root.glob("*/*/*/manifest.json"),
            *self.root.glob("*/*/*/*/manifest.json"),
        }
        for path in sorted(paths):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if document.get("schema_version") == DATASET_MANIFEST_SCHEMA and document.get("state") == "ready":
                yield path, document

    def list(self) -> list[dict[str, object]]:
        return [
            {
                "dataset_id": document["dataset_id"],
                "logical_name": document["logical_name"],
                "provider": document["provider"],
                "dataset_version": self._manifest_version(path, document),
                "source_revision": document["source_revision"],
                "preparation_format": document["preparation"]["format"],
                "path": self._display(path.parent),
                "state": document["state"],
            }
            for path, document in self._manifests()
        ]

    def _find(self, dataset_id: str) -> tuple[Path, dict[str, object]]:
        for path, document in self._manifests():
            if document.get("dataset_id") == dataset_id:
                return path.parent, document
        raise DatasetUnavailableError(f"dataset is not ready in the local registry: {dataset_id}")

    def inspect(self, dataset_id: str) -> dict[str, object]:
        path, document = self._find(dataset_id)
        return {
            **document,
            "dataset_version": self._manifest_version(path / "manifest.json", document),
            "host_path": self._display(path),
        }

    def find_resolution(self, resolution_id: str) -> dict[str, object] | None:
        for path, document in self._manifests():
            if document.get("resolution_id") == resolution_id:
                return {
                    **document,
                    "dataset_version": self._manifest_version(path, document),
                    "host_path": self._display(path.parent),
                }
        return None

    def fetch(self, request: DatasetRequest, *, allow_download: bool, progress: Progress | None = None) -> dict[str, object]:
        if request.source == "local":
            raise DatasetRequestError("use dataset import for a local source")
        if not allow_download:
            raise DatasetUnavailableError("dataset acquisition requires explicit --allow-dataset-download")
        resolution = self.resolve(request)
        with self._lock(resolution.resolution_id):
            ready = self.find_resolution(resolution.resolution_id)
            if ready is not None:
                self.verify(str(ready["dataset_id"]))
                return {**ready, "reused": True}
            self._assert_version_slot_available(resolution)
            return self._materialize(request, resolution, progress=progress)

    def import_local(self, request: DatasetRequest) -> dict[str, object]:
        if request.source != "local" or request.local_path is None:
            raise DatasetRequestError("local dataset import requires source=local and a path")
        resolution = self.resolve(request)
        with self._lock(resolution.resolution_id):
            ready = self.find_resolution(resolution.resolution_id)
            if ready is not None:
                self.verify(str(ready["dataset_id"]))
                return {**ready, "reused": True}
            self._assert_version_slot_available(resolution)
            return self._materialize(request, resolution)

    def _recoverable_staging(self, resolution_id: str) -> Path | None:
        staging_root = self.root / ".staging"
        if not staging_root.is_dir():
            return None
        candidates: list[Path] = []
        for candidate in staging_root.glob(f"{resolution_id}-*"):
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            state_path = candidate / "state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                state.get("resolution_id") == resolution_id
                and state.get("state") in {"failed", "interrupted"}
                and (candidate / "raw").is_dir()
                and not (candidate / "raw").is_symlink()
            ):
                candidates.append(candidate)
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _materialize(self, request: DatasetRequest, resolution: DatasetResolution, progress: Progress | None = None) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        bridge = self.registry.get(request.source)
        staging = self._recoverable_staging(resolution.resolution_id) if isinstance(bridge, LiberoDatasetBridge) else None
        resumed = staging is not None
        if staging is None:
            staging = self.root / ".staging" / f"{resolution.resolution_id}-{uuid.uuid4().hex}"
        raw = staging / "raw"
        prepared = staging / "prepared"
        state_path = staging / "state.json"
        staging.mkdir(parents=True, exist_ok=resumed)
        try:
            if resumed:
                if progress is not None:
                    progress(f"resuming verified staging for {resolution.resolution_id}")
                if prepared.exists():
                    if prepared.is_symlink() or not prepared.is_dir():
                        raise DatasetSecurityError("unsafe prepared directory in recoverable dataset staging")
                    shutil.rmtree(prepared)
                (staging / "dataset.json").unlink(missing_ok=True)
            atomic_json(state_path, {"state": "downloading", "resolution_id": resolution.resolution_id}, exclusive=False, mode=0o644)
            if isinstance(bridge, LocalDatasetBridge):
                bridge.import_path(request.local_path or Path(), raw)
            else:
                bridge.acquire(resolution, raw, progress)
            atomic_json(state_path, {"state": "verifying", "resolution_id": resolution.resolution_id}, exclusive=False, mode=0o644)
            acquired = bridge.verify(resolution, raw)
            atomic_json(state_path, {"state": "preparing", "resolution_id": resolution.resolution_id}, exclusive=False, mode=0o644)
            prepared_result = bridge.prepare(resolution, raw, prepared)
            preparation = {
                "schema_version": DATASET_PREPARATION_SCHEMA,
                "recipe": resolution.preparation_format,
                "recipe_version": bridge.bridge_version,
                "format": prepared_result["format"],
                "content_digest": prepared_result["content_digest"],
                "schema": prepared_result.get("schema", {}),
            }
            dataset_identity = {
                "provider": resolution.provider,
                "logical_name": resolution.logical_name,
                "source_revision": resolution.source_revision,
                "raw_content_digest": acquired["content_digest"],
                "preparation": preparation,
                "license": resolution.source_metadata.get("license"),
                "citation": resolution.source_metadata.get("citation"),
            }
            dataset_id = identity("dataset", dataset_identity, 32)
            dataset_version = self._storage_version(resolution)
            final = self._publication_path(resolution)
            if final.exists():
                raise DatasetIntegrityError(f"dataset publication target already exists without a reusable manifest: {final}")
            atomic_json(staging / "dataset.json", {
                "schema_version": DATASET_SOURCE_SCHEMA,
                "dataset_id": dataset_id,
                "resolution": resolution.as_dict(),
                "identity": dataset_identity,
            }, exclusive=False, mode=0o644)
            state_path.unlink(missing_ok=True)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            source_record_path = final / "raw" / "source.json"
            source_record = (
                json.loads(source_record_path.read_text(encoding="utf-8"))
                if source_record_path.is_file()
                else {"schema_version": DATASET_SOURCE_SCHEMA, "source_path_recorded": False}
            )
            manifest = {
                "schema_version": DATASET_MANIFEST_SCHEMA,
                "dataset_id": dataset_id,
                "logical_name": resolution.logical_name,
                "provider": resolution.provider,
                "dataset_version": dataset_version,
                "source_type": resolution.source_type,
                "canonical_locator": resolution.canonical_locator,
                "source_revision": resolution.source_revision,
                "resolution_id": resolution.resolution_id,
                "state": "ready",
                "expected_content_digest": resolution.expected_digest,
                "raw_content_digest": acquired["content_digest"],
                "prepared_content_digest": prepared_result["content_digest"],
                "raw_files": acquired["files"],
                "prepared_files": prepared_result["files"],
                "total_size": int(acquired["total_size"]) + int(prepared_result["total_size"]),
                "sample_count": prepared_result.get("sample_count"),
                "preparation": preparation,
                "schema": prepared_result.get("schema", {}),
                "license": resolution.source_metadata.get("license"),
                "citation": resolution.source_metadata.get("citation"),
                "bridge": {"id": bridge.bridge_id, "version": bridge.bridge_version},
                "source_metadata": dict(resolution.source_metadata),
                "acquisition": source_record,
                "acquired_at": _utc_now(),
                "failure": None,
            }
            atomic_json(final / "manifest.json", manifest)
            for path in sorted(final.rglob("*"), reverse=True):
                if path.is_file():
                    path.chmod(0o444)
                elif path.is_dir():
                    path.chmod(0o555)
            final.chmod(0o555)
            return {**manifest, "host_path": self._display(final), "reused": False}
        except KeyboardInterrupt as exc:
            atomic_json(state_path, {
                "state": "interrupted", "resolution_id": resolution.resolution_id,
                "failure": {"type": type(exc).__name__, "message": "operation interrupted"},
            }, exclusive=False, mode=0o644)
            raise DatasetInterruptedError(f"dataset acquisition interrupted; evidence preserved at {staging}") from exc
        except Exception as exc:
            if staging.exists():
                atomic_json(state_path, {
                    "state": "failed", "resolution_id": resolution.resolution_id,
                    "failure": {"type": type(exc).__name__, "message": str(exc)[:2000]},
                }, exclusive=False, mode=0o644)
            raise

    def verify(self, dataset_id: str) -> dict[str, object]:
        root, manifest = self._find(dataset_id)
        for section, base in (("raw_files", root / "raw"), ("prepared_files", root / "prepared")):
            for entry in manifest.get(section, ()):
                relative = safe_relative(str(entry["path"]))
                path = base / relative
                if not path.is_file() or path.is_symlink():
                    raise DatasetIntegrityError(f"dataset file is missing or unsafe: {section}/{relative}")
                if path.stat().st_size != int(entry["size"]) or sha256_file(path) != entry["sha256"]:
                    raise DatasetIntegrityError(f"dataset file failed integrity verification: {section}/{relative}")
        raw_files, raw_digest, _ = inventory(root / "raw", exclude=("source.json",))
        expected_raw = manifest["raw_content_digest"]
        # URL manifests inventory only their payload and LIBERO/local include source metadata separately.
        if raw_digest != expected_raw:
            bridge = manifest.get("bridge", {}).get("id") if isinstance(manifest.get("bridge"), dict) else None
            if bridge != "url":
                raise DatasetIntegrityError(f"dataset raw aggregate digest mismatch: expected {expected_raw}, got {raw_digest}")
        _, prepared_digest, _ = inventory(root / "prepared")
        if prepared_digest != manifest["prepared_content_digest"]:
            raise DatasetIntegrityError(
                f"dataset prepared aggregate digest mismatch: expected {manifest['prepared_content_digest']}, got {prepared_digest}"
            )
        return {
            "schema_version": "ovlab.dataset-verification/v1",
            "dataset_id": dataset_id,
            "status": "verified",
            "raw_content_digest": manifest["raw_content_digest"],
            "prepared_content_digest": manifest["prepared_content_digest"],
            "verified_file_count": len(manifest.get("raw_files", ())) + len(manifest.get("prepared_files", ())),
            "host_path": self._display(root),
        }

    def prepare(self, dataset_id: str, preparation_format: str) -> dict[str, object]:
        root, manifest = self._find(dataset_id)
        current = manifest.get("preparation", {}).get("format") if isinstance(manifest.get("preparation"), dict) else None
        if current != preparation_format:
            raise DatasetRequestError(
                f"dataset {dataset_id} is immutable and prepared as {current!r}; acquire/import a new build for {preparation_format!r}"
            )
        result = self.verify(dataset_id)
        return {**result, "preparation_format": current, "reused": True, "host_path": self._display(root)}
