"""Canonical identities and filesystem primitives for Gate I artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def identity(prefix: str, document: object, length: int = 24) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json(document)).hexdigest()[:length]}"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"unsafe relative path: {path}")
    return value


def inventory(root: Path, *, exclude: Iterable[str] = ()) -> tuple[list[dict[str, object]], str, int]:
    root = root.resolve()
    excluded = set(exclude)
    files: list[dict[str, object]] = []
    total = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "sha256": sha256_file(path)})
        total += size
    digest = hashlib.sha256(canonical_json(files)).hexdigest()
    return files, digest, total


def atomic_json(path: Path, document: Mapping[str, object], *, exclusive: bool = True, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(document)
    temporary = path.with_name(f".{path.name}.{next(tempfile._get_candidate_names())}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        path.chmod(mode)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.expanduser().resolve()):
        raise ValueError(f"{label} escapes configured root: {path}")
    return resolved


def redact_url(value: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))
