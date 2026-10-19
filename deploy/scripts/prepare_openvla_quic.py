#!/usr/bin/env python3
"""Verify the published QuIC submodule or reconstruct its historical patch."""

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
    base_revision = snapshot["base_revision"]
    published_revision = snapshot["published_revision"]
    if revision not in {base_revision, published_revision}:
        raise RuntimeError(
            "QuIC submodule revision is neither the historical migration base nor "
            "the published revision; run git submodule update --init external/openvla-quic"
        )

    def differences():
        changed = [name for name, digest in snapshot["patched_files"].items()
                   if not (repository / name).is_file()
                   or hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest]
        changed.extend(name for name in snapshot.get("deleted_files", []) if (repository / name).exists())
        return changed

    changed = differences()
    if revision == published_revision:
        if changed:
            raise RuntimeError("published QuIC submodule files are missing or changed: " + ", ".join(changed))
        status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        ).strip()
        if status:
            raise RuntimeError("published QuIC submodule has local changes; inspect external/openvla-quic")
        return {"status": "verified", "revision": revision, "source": "published-submodule",
                "patch_sha256": snapshot["patch_sha256"]}

    if changed and apply:
        check = subprocess.run(["git", "-C", str(repository), "apply", "--check", str(patch)], capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError("QuIC patch does not apply cleanly; preserve local edits and inspect external/openvla-quic before retrying. " + check.stderr.strip())
        subprocess.run(["git", "-C", str(repository), "apply", str(patch)], check=True)
        changed = differences()
    if changed:
        raise RuntimeError("QuIC migration files are missing or changed: " + ", ".join(changed)
                           + ". On the historical base revision, run python "
                           "deploy/scripts/prepare_openvla_quic.py --apply")
    return {"status": "verified", "revision": revision, "source": "historical-patch",
            "patch_sha256": snapshot["patch_sha256"]}


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
