#!/usr/bin/env python3
"""Emit deterministic source provenance for an OVLAB Docker build context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXCLUDED_PREFIXES = (".git/", "runs/", "checkpoints/", "datasets/", "lab/reports/")


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _included(relative: str) -> bool:
    return not relative.startswith(EXCLUDED_PREFIXES) and relative != ".git"


def build_manifest(root: Path) -> dict[str, object]:
    root = root.resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    status = tuple(line for line in _git(root, "status", "--short", "--untracked-files=all").splitlines() if line)
    tracked = _git(root, "ls-files", "-z").split("\0")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    paths = sorted({item for item in (*tracked, *untracked) if item and _included(item)})
    submodule_rows = []
    submodule_paths: set[str] = set()
    for line in _git(root, "submodule", "status", "--recursive").splitlines():
        if not line:
            continue
        state, rest = line[0], line[1:]
        revision, path, *_ = rest.split()
        submodule_paths.add(path)
        dirty = bool(_git(root / path, "status", "--short", check=False).strip())
        submodule_rows.append({"path": path, "revision": revision, "state": state, "dirty": dirty})

    files = []
    for relative in paths:
        if relative in submodule_paths:
            continue
        path = root / relative
        if path.is_file() and not path.is_symlink():
            files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha(path)})
        elif path.is_symlink():
            files.append({"path": relative, "symlink": path.readlink().as_posix()})
    payload = {
        "schema_version": "ovlab-source-manifest/1.0.0",
        "repository_revision": head,
        "dirty": bool(status) or any(row["dirty"] for row in submodule_rows),
        "status": list(status),
        "submodules": sorted(submodule_rows, key=lambda row: row["path"]),
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["source_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = json.dumps(build_manifest(args.root), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(document, end="")
    else:
        args.output.write_text(document, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
