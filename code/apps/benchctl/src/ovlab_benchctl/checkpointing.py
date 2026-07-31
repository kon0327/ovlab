"""Dependency-light host resolution for immutable policy checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from .schema import validate
from .strict_yaml import load


_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_DOWNLOAD_REPORT_BYTES = 256 * 1024 * 1024

ProgressCallback = Callable[[str], None]


class CheckpointResolutionError(RuntimeError):
    """A checkpoint could not be resolved or failed immutable verification."""


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    resource_id: str
    repo_id: str
    revision: str
    expected_sha256: str
    files: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        if _RESOURCE_ID.fullmatch(self.resource_id) is None or not self.repo_id:
            raise CheckpointResolutionError(
                "checkpoint ID must be path-safe and repository must be non-empty"
            )
        if _REVISION.fullmatch(self.revision) is None:
            raise CheckpointResolutionError(
                f"checkpoint {self.resource_id!r} requires a pinned 40-character revision"
            )
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise CheckpointResolutionError(
                f"checkpoint {self.resource_id!r} requires an aggregate SHA-256"
            )
        if not self.files:
            raise CheckpointResolutionError(
                f"checkpoint {self.resource_id!r} requires a verifiable file manifest"
            )
        manifest_lines = []
        for relative, size, digest in self.files:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not relative:
                raise CheckpointResolutionError(f"unsafe checkpoint manifest path: {relative}")
            if type(size) is not int or size <= 0:
                raise CheckpointResolutionError(f"checkpoint size must be positive: {relative}")
            if _SHA256.fullmatch(digest) is None:
                raise CheckpointResolutionError(f"invalid checkpoint file SHA-256: {relative}")
            manifest_lines.append(f"{relative} {size} {digest}\n")
        aggregate = hashlib.sha256("".join(manifest_lines).encode("utf-8")).hexdigest()
        if aggregate != self.expected_sha256:
            raise CheckpointResolutionError(
                f"checkpoint {self.resource_id!r} registry aggregate SHA-256 is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    spec: CheckpointSpec
    host_path: Path
    container_path: str
    source_kind: str
    verified_file_count: int
    materialized_without_copy: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.spec.resource_id,
            "repo_id": self.spec.repo_id,
            "revision": self.spec.revision,
            "expected_sha256": self.spec.expected_sha256,
            "host_path": str(self.host_path),
            "container_path": self.container_path,
            "source_kind": self.source_kind,
            "verified_file_count": self.verified_file_count,
            "materialized_without_copy": self.materialized_without_copy,
        }


@dataclass(frozen=True, slots=True)
class ResolvedTrainingCheckpoint:
    """Verified base-plus-artifact handoff for a finalized Gate I checkpoint."""

    checkpoint_id: str
    kind: str
    bundle_host_path: Path
    bundle_container_path: str
    base: ResolvedCheckpoint
    merge_status: str
    expected_loader: str
    training_run_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "kind": self.kind,
            "bundle_host_path": str(self.bundle_host_path),
            "bundle_container_path": self.bundle_container_path,
            "base_checkpoint": self.base.as_dict(),
            "merge_status": self.merge_status,
            "expected_loader": self.expected_loader,
            "training_run_id": self.training_run_id,
            "dataset_required": False,
            "training_runtime_started": False,
        }


def _merge(parent: dict[str, object], child: dict[str, object]) -> dict[str, object]:
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CheckpointResolutionError(f"{label} escapes the OVLAB configuration root: {path}")
    return resolved


def _composed(path: Path, config_root: Path, expected_kind: str, stack=()) -> dict[str, object]:
    path = _inside(path, config_root, expected_kind)
    if path in stack:
        raise CheckpointResolutionError(f"configuration inheritance cycle at {path}")
    document = load(path)
    if document.get("kind") != expected_kind:
        raise CheckpointResolutionError(f"{path} is not a {expected_kind} configuration")
    parent = document.get("extends")
    if parent is None:
        return document
    if not isinstance(parent, str) or not parent:
        raise CheckpointResolutionError(f"{path}.extends must be a non-empty relative path")
    return _merge(_composed(path.parent / parent, config_root, expected_kind, stack + (path,)), document)


def checkpoint_spec(repository_root: Path, experiment: str | Path) -> CheckpointSpec:
    """Resolve the selected policy's logical checkpoint identity without heavy imports."""
    root = repository_root.resolve()
    config_root = root / "configs"
    experiment_path = Path(experiment)
    if not experiment_path.is_absolute():
        experiment_path = root / experiment_path
    experiment_doc = _composed(experiment_path, config_root, "experiment")
    try:
        policy_ref = experiment_doc["components"]["policy"]  # type: ignore[index]
        registry_ref = experiment_doc["resources"]["registry"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise CheckpointResolutionError("experiment lacks policy or resource registry reference") from exc
    policy = _composed(config_root / str(policy_ref), config_root, "policy")
    try:
        resource_id = str(policy["settings"]["checkpoint_id"])  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise CheckpointResolutionError("selected policy lacks settings.checkpoint_id") from exc
    registry = _composed(config_root / str(registry_ref), config_root, "resource_registry")
    try:
        entry = registry["checkpoints"][resource_id]  # type: ignore[index]
        artifact_files = entry.get("files") or entry["artifact"]["files"]
        files = tuple(
            (str(name), int(identity["size"]), str(identity["sha256"]))
            for name, identity in artifact_files.items()
        )
        return CheckpointSpec(
            resource_id=resource_id,
            repo_id=str(entry["repo_id"]),
            revision=str(entry["revision"]),
            expected_sha256=str(entry["expected_sha256"]),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointResolutionError(
            f"registry entry {resource_id!r} lacks a complete immutable artifact identity"
        ) from exc


def checkpoint_spec_by_id(repository_root: Path, resource_id: str) -> CheckpointSpec:
    """Resolve one portable registry checkpoint without an experiment wrapper."""
    root = repository_root.resolve()
    registry = _composed(root / "configs" / "resources" / "registry.yaml", root / "configs", "resource_registry")
    try:
        entry = registry["checkpoints"][resource_id]  # type: ignore[index]
        artifact_files = entry.get("files") or entry["artifact"]["files"]
        files = tuple(
            (str(name), int(file_identity["size"]), str(file_identity["sha256"]))
            for name, file_identity in artifact_files.items()
        )
        return CheckpointSpec(
            resource_id=resource_id,
            repo_id=str(entry["repo_id"]),
            revision=str(entry["revision"]),
            expected_sha256=str(entry["expected_sha256"]),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointResolutionError(
            f"registry entry {resource_id!r} lacks a complete immutable artifact identity"
        ) from exc


def local_checkpoint_override(profile: str | Path | None, resource_id: str) -> Path | None:
    if profile is None:
        return None
    profile_path = Path(profile).expanduser().resolve()
    if not profile_path.is_file():
        raise CheckpointResolutionError(f"local profile does not exist: {profile_path}")
    document = load(profile_path)
    validate(document, str(profile_path), "local_profile")
    try:
        value = document.get("resources", {}).get("checkpoints", {}).get(resource_id, {}).get("local_path")
    except AttributeError as exc:
        raise CheckpointResolutionError("local profile checkpoint resources must be mappings") from exc
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise CheckpointResolutionError(
            f"local checkpoint override for {resource_id!r} must be an absolute path"
        )
    return path.resolve()


def _hf_snapshot(cache_root: Path, spec: CheckpointSpec) -> Path:
    hub = cache_root / "hub" if (cache_root / "hub").is_dir() else cache_root
    repository = "models--" + spec.repo_id.replace("/", "--")
    return hub / repository / "snapshots" / spec.revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def verify_checkpoint(
    spec: CheckpointSpec,
    snapshot: Path,
    *,
    progress: ProgressCallback | None = None,
) -> int:
    root = snapshot.resolve()
    if not root.is_dir():
        raise CheckpointResolutionError(f"checkpoint snapshot does not exist: {root}")
    manifest_lines = []
    for index, (relative, expected_size, expected_digest) in enumerate(spec.files, start=1):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise CheckpointResolutionError(f"unsafe checkpoint manifest path: {relative}")
        candidate = root / path
        if not candidate.is_file():
            raise CheckpointResolutionError(f"checkpoint file is missing: {relative}")
        actual_size = candidate.stat().st_size
        if actual_size != expected_size:
            raise CheckpointResolutionError(
                f"checkpoint size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
        actual_digest = _sha256(candidate)
        if actual_digest != expected_digest:
            raise CheckpointResolutionError(
                f"checkpoint SHA-256 mismatch for {relative}: expected {expected_digest}, got {actual_digest}"
            )
        _notify(
            progress,
            f"Verified [{index}/{len(spec.files)}] {relative} ({_human_bytes(actual_size)}).",
        )
        manifest_lines.append(f"{relative} {expected_size} {expected_digest}\n")
    aggregate = hashlib.sha256("".join(manifest_lines).encode("utf-8")).hexdigest()
    if aggregate != spec.expected_sha256:
        raise CheckpointResolutionError(
            f"checkpoint aggregate SHA-256 mismatch: expected {spec.expected_sha256}, got {aggregate}"
        )
    return len(spec.files)


def _link_snapshot(source: Path, destination: Path) -> None:
    """Create a self-contained snapshot using hard links, never weight copies."""
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for source_path in sorted(source.rglob("*")):
            relative = source_path.relative_to(source)
            target = temporary / relative
            if source_path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not source_path.is_file():
                raise CheckpointResolutionError(f"unsupported checkpoint entry: {source_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path.resolve(), target)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise CheckpointResolutionError(
                        "global Hugging Face cache and OVLAB managed storage are on different filesystems; "
                        "cannot reuse the snapshot without copying"
                    ) from exc
                raise
        temporary.chmod(0o755)
        for directory in (path for path in temporary.rglob("*") if path.is_dir()):
            directory.chmod(0o755)
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _has_external_symlinks(snapshot: Path) -> bool:
    root = snapshot.resolve()
    for path in snapshot.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            return True
    return False


def _same_manifest_files(spec: CheckpointSpec, first: Path, second: Path) -> bool:
    try:
        return all(
            os.path.samefile(first / relative, second / relative)
            for relative, _size, _digest in spec.files
        )
    except (FileNotFoundError, OSError):
        return False


def _prepare_mount_snapshot(snapshot: Path) -> None:
    """Make managed directories traversable without changing hard-linked cache files."""
    snapshot.chmod(0o755)
    for path in snapshot.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file() and not (path.stat().st_mode & 0o004):
            raise CheckpointResolutionError(
                f"checkpoint file is not readable by the non-root policy container: {path}"
            )


def _download_snapshot(
    repo_id: str,
    revision: str,
    destination: Path,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Download one immutable Hugging Face revision using only the Python standard library."""
    headers = {"User-Agent": "OpenVLABenchmark/0.1 checkpoint-resolver"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api = (
        f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}/revision/"
        f"{quote(revision)}"
    )
    _notify(progress, f"Querying Hugging Face metadata for {repo_id}@{revision}.")
    try:
        with urlopen(Request(api, headers=headers), timeout=60) as response:
            metadata = json.load(response)
    except Exception as exc:
        raise CheckpointResolutionError(
            f"failed to query Hugging Face checkpoint {repo_id}@{revision}"
        ) from exc
    siblings = metadata.get("siblings")
    if metadata.get("sha") != revision:
        raise CheckpointResolutionError(
            f"Hugging Face resolved {repo_id}@{revision} to unexpected revision {metadata.get('sha')!r}"
        )
    filenames = sorted(
        item.get("rfilename") for item in siblings or []
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
    )
    if not filenames:
        raise CheckpointResolutionError(f"Hugging Face returned no files for {repo_id}@{revision}")
    declared_sizes = {
        item["rfilename"]: item.get("size")
        for item in siblings or []
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
    }
    _notify(progress, f"Downloading {len(filenames)} checkpoint files from {repo_id}.")
    for index, filename in enumerate(filenames, start=1):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointResolutionError(f"Hugging Face returned unsafe path: {filename}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision)}/{quote(filename, safe='/')}?download=true"
        )
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response, target.open("xb") as stream:
                header_size = response.headers.get("Content-Length")
                declared_size = declared_sizes.get(filename)
                total = (
                    int(header_size)
                    if header_size is not None and str(header_size).isdigit()
                    else declared_size if type(declared_size) is int and declared_size >= 0
                    else None
                )
                total_label = "unknown size" if total is None else _human_bytes(total)
                _notify(progress, f"[{index}/{len(filenames)}] {filename}: starting ({total_label}).")
                downloaded = 0
                next_report = _DOWNLOAD_REPORT_BYTES
                while True:
                    block = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    stream.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        if total is None:
                            detail = _human_bytes(downloaded)
                        else:
                            percentage = min(100.0, downloaded * 100.0 / total) if total else 100.0
                            detail = (
                                f"{_human_bytes(downloaded)} / {_human_bytes(total)} "
                                f"({percentage:.1f}%)"
                            )
                        _notify(progress, f"[{index}/{len(filenames)}] {filename}: {detail}.")
                        next_report = downloaded + _DOWNLOAD_REPORT_BYTES
                if total is not None and downloaded != total:
                    raise CheckpointResolutionError(
                        f"downloaded size mismatch for {filename}: expected {total}, got {downloaded}"
                    )
                _notify(
                    progress,
                    f"[{index}/{len(filenames)}] {filename}: {_human_bytes(downloaded)} complete.",
                )
        except Exception as exc:
            if isinstance(exc, CheckpointResolutionError):
                raise
            raise CheckpointResolutionError(f"failed to download checkpoint file {filename}") from exc


class CheckpointResolver:
    """Apply deterministic source precedence and return one verified mount source."""

    def __init__(
        self,
        *,
        global_cache: Path,
        managed_cache: Path,
        downloader: Callable[[str, str, Path], None] | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.global_cache = global_cache.expanduser().resolve()
        self.managed_cache = managed_cache.expanduser().resolve()
        self.downloader = downloader
        self.progress = progress

    def resolve(
        self,
        spec: CheckpointSpec,
        *,
        local_path: Path | None = None,
        offline: bool = False,
    ) -> ResolvedCheckpoint:
        _notify(
            self.progress,
            f"Resolving checkpoint {spec.resource_id!r} ({spec.repo_id}@{spec.revision}).",
        )
        managed_snapshot = self.managed_cache / spec.resource_id / spec.revision
        candidates = (
            ("global-huggingface-cache", _hf_snapshot(self.global_cache, spec)),
            ("local-profile", local_path),
            ("ovlab-managed-cache", managed_snapshot),
        )
        for source_kind, candidate in candidates:
            if candidate is None or not candidate.is_dir():
                continue
            needs_materialization = (
                source_kind != "ovlab-managed-cache" or _has_external_symlinks(candidate)
            )
            managed_is_same = (
                needs_materialization
                and managed_snapshot.is_dir()
                and _same_manifest_files(spec, candidate, managed_snapshot)
            )
            verification_source = managed_snapshot if managed_is_same else candidate
            _notify(
                self.progress,
                f"Found checkpoint in {source_kind}; verifying {len(spec.files)} files.",
            )
            count = verify_checkpoint(spec, verification_source, progress=self.progress)
            materialized = False
            selected = candidate.resolve()
            if needs_materialization and source_kind == "ovlab-managed-cache":
                raise CheckpointResolutionError(
                    f"managed checkpoint contains symlinks outside its snapshot: {candidate}"
                )
            if needs_materialization:
                self.managed_cache.mkdir(parents=True, exist_ok=True)
                lock_path = self.managed_cache / ".resolve.lock"
                with lock_path.open("a+b") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    if not managed_snapshot.exists():
                        managed_snapshot.parent.mkdir(parents=True, exist_ok=True)
                        _link_snapshot(candidate, managed_snapshot)
                    elif not managed_is_same:
                        verify_checkpoint(spec, managed_snapshot, progress=self.progress)
                selected = managed_snapshot.resolve()
                materialized = True
            _prepare_mount_snapshot(selected)
            _notify(self.progress, f"Checkpoint {spec.resource_id!r} is verified and ready.")
            return ResolvedCheckpoint(
                spec=spec,
                host_path=selected,
                container_path=f"/checkpoints/resolved/{spec.resource_id}",
                source_kind=source_kind,
                verified_file_count=count,
                materialized_without_copy=materialized,
            )
        if offline:
            raise CheckpointResolutionError(
                f"checkpoint {spec.resource_id!r} is unavailable in global cache, local profile, "
                "and OVLAB managed cache while --offline is active"
            )
        self.managed_cache.mkdir(parents=True, exist_ok=True)
        lock_path = self.managed_cache / ".download.lock"
        with lock_path.open("a+b") as lock:
            _notify(self.progress, "Waiting for the managed checkpoint download lock.")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not managed_snapshot.exists():
                managed_snapshot.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(
                    prefix=f".{spec.revision}.", dir=managed_snapshot.parent
                ))
                try:
                    _notify(
                        self.progress,
                        f"Checkpoint {spec.resource_id!r} was not found locally; starting download.",
                    )
                    if self.downloader is None:
                        _download_snapshot(
                            spec.repo_id,
                            spec.revision,
                            temporary,
                            progress=self.progress,
                        )
                    else:
                        self.downloader(spec.repo_id, spec.revision, temporary)
                    _notify(
                        self.progress,
                        f"Download complete; verifying {len(spec.files)} files and SHA-256 hashes.",
                    )
                    count = verify_checkpoint(spec, temporary, progress=self.progress)
                    os.rename(temporary, managed_snapshot)
                except Exception:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
            else:
                _notify(
                    self.progress,
                    "Another resolver populated the managed checkpoint; verifying it now.",
                )
                count = verify_checkpoint(spec, managed_snapshot, progress=self.progress)
        _prepare_mount_snapshot(managed_snapshot)
        _notify(self.progress, f"Checkpoint {spec.resource_id!r} is verified and ready.")
        return ResolvedCheckpoint(
            spec=spec,
            host_path=managed_snapshot.resolve(),
            container_path=f"/checkpoints/resolved/{spec.resource_id}",
            source_kind="ovlab-managed-download",
            verified_file_count=count,
            materialized_without_copy=False,
        )


def default_global_cache(environment: Mapping[str, str]) -> Path:
    value = (
        environment.get("HF_HUB_CACHE")
        or environment.get("HUGGINGFACE_HUB_CACHE")
        or environment.get("HF_HOME")
    )
    return Path(value).expanduser() if value else Path.home() / ".cache/huggingface"


def resolve_finalized_training_checkpoint(
    repository_root: Path,
    model_data_root: Path,
    checkpoint_id: str,
    *,
    global_cache: Path | None = None,
    local_base_path: Path | None = None,
    progress: ProgressCallback | None = None,
) -> ResolvedTrainingCheckpoint:
    """Resolve an immutable training bundle and its pinned base without network.

    This is the deployment handoff boundary: it accepts only content-derived
    finalized IDs, independently verifies the bundle, and delegates base-model
    resolution to the pre-existing checkpoint resolver in strict offline mode.
    It never examines training staging directories or starts a trainer.
    """
    if re.fullmatch(r"checkpoint-[0-9a-f]{32}", checkpoint_id) is None:
        raise CheckpointResolutionError(
            "trained checkpoint selection requires an immutable checkpoint-<32 hex> ID; aliases are not accepted"
        )
    # Local import avoids coupling the existing Hugging Face resolver to the
    # training domain at module import time.
    from .training_runs import CheckpointBundleStore

    root = model_data_root.expanduser().resolve()
    store = CheckpointBundleStore(root)
    try:
        store.verify(checkpoint_id)
        inspection = store.inspect(checkpoint_id)
    except Exception as exc:
        raise CheckpointResolutionError(
            f"trained checkpoint is not finalized and verified: {checkpoint_id}: {exc}"
        ) from exc
    checkpoint = inspection.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise CheckpointResolutionError("trained checkpoint document is missing")
    kind = str(checkpoint.get("kind"))
    if kind not in {"peft_adapter", "full_checkpoint"}:
        raise CheckpointResolutionError(f"unsupported trained checkpoint kind: {kind!r}")
    if kind == "peft_adapter" and checkpoint.get("merge_status") != "unmerged":
        raise CheckpointResolutionError("Gate I PEFT deployment requires an unmerged adapter bundle")
    base_identity = checkpoint.get("base_checkpoint")
    if not isinstance(base_identity, dict):
        raise CheckpointResolutionError("trained checkpoint omits its immutable base dependency")
    resource_id = str(base_identity.get("resource_id", ""))
    try:
        spec = checkpoint_spec_by_id(repository_root, resource_id)
    except CheckpointResolutionError as exc:
        raise CheckpointResolutionError(
            f"trained checkpoint base dependency is incompatible with registry resource {resource_id!r}: {exc}"
        ) from exc
    expected = {
        "resource_id": spec.resource_id,
        "repository": spec.repo_id,
        "revision": spec.revision,
        "aggregate_sha256": spec.expected_sha256,
    }
    if any(base_identity.get(key) != value for key, value in expected.items()):
        raise CheckpointResolutionError(
            f"trained checkpoint base dependency is incompatible with registry resource {resource_id!r}"
        )
    resolver = CheckpointResolver(
        global_cache=(global_cache or default_global_cache(os.environ)).expanduser().resolve(),
        managed_cache=root / "checkpoints" / "huggingface",
        progress=progress,
    )
    base = resolver.resolve(spec, local_path=local_base_path, offline=True)
    bundle_path = Path(str(inspection["host_path"])).resolve()
    return ResolvedTrainingCheckpoint(
        checkpoint_id=checkpoint_id,
        kind=kind,
        bundle_host_path=bundle_path,
        bundle_container_path=f"/checkpoints/trained/{checkpoint_id}",
        base=base,
        merge_status=str(checkpoint.get("merge_status")),
        expected_loader=str(checkpoint.get("expected_loader")),
        training_run_id=str(checkpoint.get("training_run_id")),
    )
