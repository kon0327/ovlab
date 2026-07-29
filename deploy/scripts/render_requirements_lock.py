#!/usr/bin/env python3
"""Render pip --require-hashes input from a PEP 751 pylock file."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 lock-generation environment
    import tomli as tomllib


BASE_PROVIDED = {"torch", "torchvision", "triton"}


def render(source: Path) -> str:
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    lines = [
        "# Generated from " + source.name + "; do not edit.",
        "# Torch, TorchVision, Triton and CUDA libraries are supplied by the pinned base image.",
    ]
    for package in document["packages"]:
        name = package["name"]
        if name in BASE_PROVIDED or name.startswith("nvidia-"):
            continue
        if "vcs" in package:
            vcs = package["vcs"]
            lines.append(
                f"# VCS: {name} @ git+{vcs['url']}@{vcs['commit-id']} "
                "(installed separately at the immutable commit)"
            )
            continue
        artifact = package.get("wheels", [package.get("sdist")])[0]
        digest = artifact["hashes"]["sha256"]
        lines.append(f"{name}=={package['version']} --hash=sha256:{digest}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.source), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
