"""Dependency-free CLI version and source-revision helpers."""

from pathlib import Path
import os
import re

CLI_VERSION = "0.2.0"


def repository_revision(repository_root: str | Path) -> str | None:
    embedded = os.environ.get("OVLAB_REVISION")
    if embedded is not None:
        return embedded if re.fullmatch(r"[0-9a-f]{40}", embedded) else None
    root = Path(repository_root)
    try:
        value = (root / ".git/HEAD").read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            value = (root / ".git" / value.removeprefix("ref: ")).read_text(encoding="utf-8").strip()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    except OSError:
        return None
