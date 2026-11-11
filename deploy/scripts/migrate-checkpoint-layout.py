#!/usr/bin/env python3
"""Verify and optionally move legacy training bundles into checkpoints/local."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
for parent in (REPOSITORY / "code/packages", REPOSITORY / "code/policies", REPOSITORY / "code/apps"):
    for src in parent.glob("*/src"):
        sys.path.insert(0, str(src))

from ovlab_benchctl.training_runs import CheckpointBundleStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="perform the verified moves")
    args = parser.parse_args()
    root = args.model_data_root.expanduser().resolve() / "checkpoints"
    if not root.is_dir() or root.is_symlink():
        parser.error(f"checkpoint root does not exist or is a symlink: {root}")
    local, others = root / "local", root / "others"
    if local.is_symlink() or others.is_symlink():
        parser.error("local and others must not be symbolic links")
    store = CheckpointBundleStore(root.parent)
    sources = sorted(path for path in root.iterdir() if re.fullmatch(r"checkpoint-[0-9a-f]{32}", path.name))
    for source in sources:
        if source.is_symlink() or not source.is_dir():
            parser.error(f"legacy checkpoint is not a directory: {source}")
        if (local / source.name).exists() or (local / source.name).is_symlink():
            parser.error(f"destination already exists: {local / source.name}")
        store.verify(source.name)
        print(f"{source} -> {local / source.name}")
    print(f"Verified {len(sources)} legacy checkpoint(s).")
    if not args.apply:
        print("Dry run only; pass --apply to move them.")
        return 0
    if not os.access(root, os.W_OK) or any(not os.access(source, os.W_OK) for source in sources):
        parser.error("insufficient write permission on checkpoint root or sealed checkpoint directories; no files were moved")
    if local.exists() and not os.access(local, os.W_OK):
        parser.error(f"checkpoint destination is not writable: {local}")
    created_local, created_others = not local.exists(), not others.exists()
    local.mkdir(exist_ok=True)
    others.mkdir(exist_ok=True)
    # New folders should inherit the data owner's identity even if migration is
    # run by an administrator to move sealed, container-owned directories.
    if os.geteuid() == 0:
        owner = root.stat()
        for directory, created in ((local, created_local), (others, created_others)):
            if created:
                os.chown(directory, owner.st_uid, owner.st_gid)
    if local.stat().st_uid == os.geteuid() or os.geteuid() == 0:
        local.chmod(0o2770)
    moved = []
    try:
        for source in sources:
            destination = local / source.name
            source.rename(destination)
            moved.append((source, destination))
            store.verify(source.name)
    except Exception:
        for source, destination in reversed(moved):
            destination.rename(source)
        raise
    print(f"Moved and verified {len(moved)} checkpoint(s); created {others}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
