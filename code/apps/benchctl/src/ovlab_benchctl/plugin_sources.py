"""Fetch pinned plugin sources into an ignored, reproducible build workspace."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .errors import ConfigReferenceError, ConfigCompatibilityError
from .policy_plugins import registry_document

_EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ovlab-source.json"}
_GENERATED_METADATA_SUFFIXES = (".egg-info", ".dist-info")


def _excluded_source_path(relative):
    """Match generated source artifacts excluded from the Docker context."""
    path = Path(relative)
    if any(part in _EXCLUDED or part.endswith(_GENERATED_METADATA_SUFFIXES) for part in path.parts):
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return "rollouts" in path.parts[:-1] and path.suffix == ".mp4"


def _git(path, *args, check=True):
    result = subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True)
    if check and result.returncode:
        raise ConfigReferenceError(f"Git source operation failed: {result.stderr.strip()}")
    return result.stdout.strip()


def tree_identity(path):
    """Hash paths, modes, symlink targets and all regular source bytes."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ConfigReferenceError(f"source directory does not exist: {root}")
    digest = hashlib.sha256()
    count = 0
    def unreadable(error):
        raise ConfigReferenceError(f"cannot inspect source directory: {error.filename}") from error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=unreadable):
        parent = Path(directory).relative_to(root)
        dirs[:] = sorted(d for d in dirs if not _excluded_source_path(parent / d))
        names = sorted(files + [d for d in dirs if (Path(directory) / d).is_symlink()])
        for name in names:
            file = Path(directory) / name
            if _excluded_source_path(file.relative_to(root)): continue
            relative = file.relative_to(root).as_posix()
            digest.update(relative.encode() + b"\0")
            if file.is_symlink():
                if not file.resolve().is_relative_to(root):
                    raise ConfigReferenceError(f"source symlink escapes its project: {file}")
                digest.update(b"link\0" + os.readlink(file).encode())
            elif file.is_file():
                digest.update(str(file.stat().st_mode & 0o111).encode() + b"\0")
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
            else:
                raise ConfigReferenceError(f"source contains unsupported file: {file}")
            digest.update(b"\0")
            count += 1
    return {"content_sha256": digest.hexdigest(), "file_count": count}


def _selected(root, ids, registry=None):
    sources = (registry if registry is not None else registry_document(root))["sources"]
    selected = list(sources) if ids is None else list(dict.fromkeys(ids))
    unknown = set(selected) - sources.keys()
    if unknown: raise ConfigReferenceError(f"unknown plugin sources: {', '.join(sorted(unknown))}")
    return [(name, sources[name]) for name in selected]


def managed_source_relative(name):
    if not isinstance(name, str) or "\\" in name or Path(name).is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ConfigReferenceError(f"unsafe managed source identifier: {name}")
    return Path("external/extensions" if "/" in name else "external") / name


def _destination(root, name):
    external = Path(root) / "external"
    path = Path(root) / managed_source_relative(name)
    if external.is_symlink() or any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(external)):
        raise ConfigReferenceError(f"managed source cannot be a symlink: {path}")
    return path


def inspect_sources(repository_root, source_ids=None, *, registry=None, staged=False):
    root = Path(repository_root).resolve()
    rows = []
    for name, source in _selected(root, source_ids, registry):
        path = _destination(root, name)
        row = {"id": name, "kind": source["kind"], "path": str(path), "present": path.is_dir(), "selection": dict(source)}
        if source["kind"] == "local": row["dirty"] = True
        content_path = path
        if source["kind"] == "local" and not staged:
            content_path = Path(source["path"]).expanduser()
            if not content_path.is_absolute(): content_path = root / content_path
            if not content_path.is_dir() and path.is_dir(): content_path = path
        if content_path.is_dir():
            row.update(tree_identity(content_path))
            if source["kind"] == "git":
                row.update(revision=_git(path, "rev-parse", "HEAD", check=False),
                           dirty=bool(_uncommitted_files(path)))
                row["matches_commit"] = row["revision"] == source["commit"]
        rows.append(row)
    return rows


@contextmanager
def _source_lock(root, name):
    external = Path(root) / "external"
    locks = external / ".locks"
    if external.is_symlink() or locks.is_symlink():
        raise ConfigReferenceError("managed source and lock directories cannot be symlinks")
    locks.mkdir(parents=True, exist_ok=True)
    # O_NOFOLLOW closes the final-component symlink hole before opening a lock.
    fd = os.open(locks / (hashlib.sha256(name.encode()).hexdigest() + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _uncommitted_files(path):
    """Include Git-ignored files that our source snapshot would also include."""
    changed = _git(path, "diff", "--name-only", "-z", "HEAD").split("\0")
    untracked = _git(path, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    ignored = _git(path, "ls-files", "--others", "--ignored", "--exclude-standard", "-z").split("\0")
    # Never discard tracked edits, including edits to files excluded from the
    # build: a checkout of another revision must not overwrite local work.
    return sorted({name for name in changed if name} | {
        name for name in (*untracked, *ignored) if name and not _excluded_source_path(name)
    })



def _sync_git(root, name, source, offline, git_cache_root=None):
    path = _destination(root, name)
    if path.exists():
        if not path.is_dir() or not (path / ".git").exists():
            raise ConfigCompatibilityError(f"source destination is not a Git checkout: {path}")
        origin = _git(path, "remote", "get-url", "origin")
        if origin.removesuffix(".git").rstrip("/") != source["url"].removesuffix(".git").rstrip("/"):
            raise ConfigCompatibilityError(f"source {name} origin differs from registry; preserve or move {path} before syncing")
        changes = _uncommitted_files(path)
        if changes:
            examples = ", ".join(repr(item) for item in changes[:5])
            suffix = f" (and {len(changes) - 5} more)" if len(changes) > 5 else ""
            raise ConfigCompatibilityError(
                f"source {name} has local changes: {examples}{suffix}; "
                "register it as a local source or preserve changes before syncing"
            )
        if _git(path, "rev-parse", "HEAD") == source["commit"]: return
        exists = _git(path, "cat-file", "-t", source["commit"], check=False) == "commit"
        if not exists:
            if offline: raise ConfigReferenceError(f"source {name}@{source['commit']} is unavailable offline")
            _git(path, "fetch", "--no-tags", "origin", source["commit"])
        _git(path, "checkout", "--detach", source["commit"])
        return
    fetch_from = "origin"
    if offline:
        cached = _destination(git_cache_root, name) if git_cache_root is not None else None
        if (cached is None or not (cached / ".git").exists()
                or _git(cached, "cat-file", "-t", source["commit"], check=False) != "commit"):
            raise ConfigReferenceError(f"source {name}@{source['commit']} is unavailable offline; sync it first")
        # Consume immutable objects only. Dirty files in the development checkout
        # are neither copied nor modified by an isolated offline build.
        fetch_from = str(cached)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{Path(name).name}-", dir=path.parent))
    try:
        _git(temporary, "init", "--quiet")
        _git(temporary, "remote", "add", "origin", source["url"])
        _git(temporary, "fetch", "--depth=1", "--no-tags", "--no-recurse-submodules", fetch_from, source["commit"])
        _git(temporary, "checkout", "--detach", source["commit"])
        if _git(temporary, "rev-parse", "HEAD") != source["commit"]:
            raise ConfigCompatibilityError(f"source {name} resolved to an unexpected commit")
        tree_identity(temporary)
        temporary.rename(path)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def _sync_local(root, name, source):
    original = Path(source["path"]).expanduser()
    if not original.is_absolute(): original = root / original
    original = original.resolve()
    destination = _destination(root, name)
    if not original.is_dir(): raise ConfigReferenceError(f"local source does not exist: {original}")
    if original == destination.resolve():
        tree_identity(original)
        return
    if original.is_relative_to(destination.resolve()) or destination.resolve().is_relative_to(original):
        raise ConfigReferenceError("local source and managed snapshot must not contain each other")
    identity = tree_identity(original)
    marker = destination / ".ovlab-source.json"
    if destination.exists():
        if not marker.is_file(): raise ConfigCompatibilityError(f"refusing to replace unmanaged local source {destination}")
        try:
            saved = json.loads(marker.read_text())
            if not isinstance(saved, dict) or not isinstance(saved["source"], str) or not isinstance(saved["identity"], dict):
                raise ValueError("invalid snapshot metadata")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigCompatibilityError(f"invalid local source snapshot metadata: {marker}") from exc
        if tree_identity(destination) != saved["identity"]:
            raise ConfigCompatibilityError(f"local source snapshot has changes: {destination}")
        if saved["source"] == str(original) and identity == saved["identity"]: return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{Path(name).name}-", dir=destination.parent))
    try:
        def ignore_generated(directory, names):
            relative = Path(directory).relative_to(original)
            return {item for item in names if _excluded_source_path(relative / item)}

        shutil.copytree(original, temporary, dirs_exist_ok=True, symlinks=True,
                        ignore=ignore_generated)
        if tree_identity(temporary) != identity:
            raise ConfigCompatibilityError("local source changed while creating its build snapshot")
        (temporary / ".ovlab-source.json").write_text(json.dumps({"source": str(original), "identity": identity}, sort_keys=True))
        backup = None
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{Path(name).name}-previous-", dir=destination.parent))
            backup.rmdir()
            destination.rename(backup)
        try:
            temporary.rename(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


@contextmanager
def prepared_sources(repository_root, source_ids=None, *, offline=False, registry=None, git_cache_root=None):
    """Keep the selected build workspace locked until its consumer finishes."""
    root = Path(repository_root).resolve()
    registry = registry if registry is not None else registry_document(root)
    selections = _selected(root, source_ids, registry)
    with ExitStack() as stack:
        for name, _ in sorted(selections):
            stack.enter_context(_source_lock(root, name))
        for name, source in selections:
            if source["kind"] == "git": _sync_git(root, name, source, offline, git_cache_root)
            else: _sync_local(root, name, source)
        yield inspect_sources(root, [name for name, _ in selections], registry=registry, staged=True)


def sync_sources(repository_root, source_ids=None, *, offline=False, dry_run=False):
    root = Path(repository_root).resolve()
    if dry_run:
        return [{**row, "action": "verify-or-sync", "side_effects_performed": False}
                for row in inspect_sources(root, source_ids)]
    with prepared_sources(root, source_ids, offline=offline) as rows:
        return rows
