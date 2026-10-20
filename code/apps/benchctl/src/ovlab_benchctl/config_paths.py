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

    # Paths explicitly rooted at configs/ retain their repository-relative meaning.
    if path.parts and path.parts[0] == "configs":
        return direct
    names = [path]
    if not path.suffix:
        names.append(path.with_suffix(".yaml"))
    for name in names:
        candidate = (root / "configs" / section / name).resolve()
        if candidate.is_relative_to((root / "configs" / section).resolve()) and candidate.is_file():
            return candidate
    return direct
