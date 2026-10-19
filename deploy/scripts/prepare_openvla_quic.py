#!/usr/bin/env python3
"""Verify or reconstruct the reviewed QuIC additions on the pinned submodule."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(root, *, apply=False):
    root = Path(root).resolve()
    repository = root / "external/openvla-quic"
    snapshot = json.loads((root / "external/openvla-quic.migration.json").read_text())
    patch = root / "external/openvla-quic.migration.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != snapshot["patch_sha256"]:
        raise RuntimeError("QuIC migration patch SHA256 differs from its provenance manifest")
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != snapshot["base_revision"]:
        raise RuntimeError("QuIC submodule revision differs from migration base; run git submodule update --init external/openvla-quic")

    def differences():
        return [name for name, digest in snapshot["patched_files"].items()
                if not (repository / name).is_file()
                or hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest]

    changed = differences()
    if changed and apply:
        check = subprocess.run(["git", "-C", str(repository), "apply", "--check", str(patch)], capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError("QuIC patch does not apply cleanly; preserve local edits and inspect external/openvla-quic before retrying. " + check.stderr.strip())
        subprocess.run(["git", "-C", str(repository), "apply", str(patch)], check=True)
        changed = differences()
    if changed:
        raise RuntimeError("QuIC migration files are missing or changed: " + ", ".join(changed)
                           + ". Restore the pinned submodule, then run python deploy/scripts/prepare_openvla_quic.py --apply")
    return {"status": "verified", "base_revision": revision, "patch_sha256": snapshot["patch_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.root, apply=args.apply), sort_keys=True))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
