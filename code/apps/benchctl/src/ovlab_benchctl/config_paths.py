"""Resolve convenient profile names within OVLAB's configuration tree."""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(reference: str | Path, repository_root: Path, section: str) -> Path:
    """Resolve explicit paths first, then bare YAML names in a config section."""
    path = Path(reference).expanduser()
    root = repository_root.expanduser().resolve()
    if path.is_absolute():
        return path.resolve()

    direct = (root / path).resolve()
    if direct.is_file():
        return direct

    # Only a bare filename gets the section default. Explicit relative paths
    # retain their old repository-root-relative meaning.
    if path.name != str(path):
        return direct
    names = [path]
    if not path.suffix:
        names.append(path.with_suffix(".yaml"))
    for name in names:
        candidate = (root / "configs" / section / name).resolve()
        if candidate.parent == (root / "configs" / section).resolve() and candidate.is_file():
            return candidate
    return direct
