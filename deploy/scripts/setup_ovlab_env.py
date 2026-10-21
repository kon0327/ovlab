#!/usr/bin/env python3
"""Install the host CLI in a named Conda environment or a local venv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONDA_ENV_NAME = "ovlab-env"
PACKAGES = (
    "code/packages/ovlab-core",
    "code/packages/ovlab-policy-sdk",
    "code/packages/ovlab-benchmarks",
    "code/packages/ovlab-metrics",
    "code/packages/ovlab-openvla-common",
    "code/packages/ovlab-remote-policy",
    "code/apps/runner",
    "code/policies/openvla-vanilla",
    "code/policies/openvla-lora-merged",
    "code/policies/openvla-oft",
    "code/policies/openvla-quic",
    "code/apps/benchctl",
)


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _manager(requested: str) -> str:
    manager = ("conda" if shutil.which("conda") else "venv") if requested == "auto" else requested
    if manager == "conda" and not shutil.which("conda"):
        raise RuntimeError("Conda was requested but is not on PATH; use --manager venv")
    return manager


def _named_conda_prefix(name: str) -> Path | None:
    completed = subprocess.run(
        ["conda", "env", "list", "--json"], capture_output=True, text=True, check=True,
    )
    data = json.loads(completed.stdout)
    matches = [
        Path(path) for path in data.get("envs", [])
        if data.get("envs_details", {}).get(path, {}).get("name", Path(path).name).casefold() == name.casefold()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple Conda environments are named {name}: {matches}")
    return matches[0] if matches else None


def _check_prefix(prefix: Path, manager: str) -> None:
    if not prefix.exists():
        return
    marker = prefix / ("conda-meta" if manager == "conda" else "pyvenv.cfg")
    if not marker.exists():
        raise RuntimeError(f"{prefix} exists but is not a {manager} environment; choose another --prefix")


def _python(prefix: Path) -> Path:
    if os.name == "nt":
        # Conda places python.exe at the prefix root; venv uses Scripts.
        root_python = prefix / "python.exe"
        return root_python if root_python.is_file() else prefix / "Scripts" / "python.exe"
    return prefix / "bin" / "python"


def _command(prefix: Path, manager: str, name: str | None = None) -> list[str]:
    if manager == "conda":
        selector = ["--name", name] if name else ["--prefix", str(prefix)]
        return ["conda", "run", "--no-capture-output", *selector, "python"]
    return [str(_python(prefix))]


def _check_base_python(executable: str) -> None:
    completed = subprocess.run(
        [executable, "-c", "import sys; print(*sys.version_info[:2], sep='.')"],
        capture_output=True, text=True, check=True,
    )
    major, minor = (int(part) for part in completed.stdout.strip().split("."))
    if (major, minor) < (3, 10):
        raise RuntimeError("the venv fallback requires Python 3.10 or newer; choose --manager conda")


def _requirements_fingerprint() -> str:
    digest = hashlib.sha256()
    for package in PACKAGES:
        digest.update(package.encode())
        digest.update((ROOT / package / "pyproject.toml").read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager", choices=("auto", "conda", "venv"), default="auto")
    parser.add_argument("--prefix", type=Path, help="custom environment path; otherwise Conda uses ovlab-env")
    parser.add_argument("--python", default=sys.executable, help="base Python for the venv fallback")
    parser.add_argument("--update", action="store_true", help="reinstall packages and recheck dependencies")
    args = parser.parse_args(argv)
    manager = _manager(args.manager)
    name = CONDA_ENV_NAME if manager == "conda" and args.prefix is None else None
    prefix = (args.prefix or ROOT / ".ovlab-env").expanduser().resolve() if name is None else _named_conda_prefix(name)

    if prefix is None or not prefix.exists():
        if manager == "conda":
            selector = ["--name", name] if name else ["--prefix", str(prefix)]
            if prefix is not None:
                prefix.parent.mkdir(parents=True, exist_ok=True)
            _run(["conda", "create", "--yes", *selector, "python=3.10", "pip", "setuptools>=68"])
            if name:
                prefix = _named_conda_prefix(name)
                if prefix is None:
                    raise RuntimeError(f"Conda environment {name} was created but cannot be found")
        else:
            assert prefix is not None
            prefix.parent.mkdir(parents=True, exist_ok=True)
            _check_base_python(args.python)
            _run([args.python, "-m", "venv", str(prefix)])
    assert prefix is not None
    _check_prefix(prefix, manager)
    if not _python(prefix).is_file():
        raise RuntimeError(f"environment Python was not created: {_python(prefix)}")

    python = _command(prefix, manager, name)
    marker = prefix / ".ovlab-host-ready"
    fingerprint = _requirements_fingerprint()
    if not args.update and marker.is_file() and marker.read_text().strip() == fingerprint:
        try:
            _run([*python, "-m", "ovlab_benchctl.cli", "--version"])
            print(f"OVLAB environment ready: {name or prefix} ({manager})")
            return 0
        except subprocess.CalledProcessError:
            print("Existing OVLAB CLI check failed; repairing the environment.", flush=True)

    _run([*python, "-m", "pip", "install", "--disable-pip-version-check", "setuptools>=68"])
    _run([
        *python, "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
        *(argument for package in PACKAGES for argument in ("--editable", str(ROOT / package))),
    ])
    _run([*python, "-m", "pip", "check"])
    _run([*python, "-c", "import ovlab_benchctl, wandb; print('OVLAB host imports OK; wandb', wandb.__version__)"])
    _run([*python, "-m", "ovlab_benchctl.cli", "--version"])
    marker.write_text(fingerprint + "\n")

    print(f"\nOVLAB environment ready: {name or prefix} ({manager})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"OVLAB environment setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
