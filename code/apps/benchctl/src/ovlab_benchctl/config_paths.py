"""Resolve convenient profile names within OVLAB's configuration tree."""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(reference: str | Path, repository_root: Path, section: str) -> Path:
    """Resolve explicit paths first, then bare YAML names in a config section."""
    from .policy_plugins import plugin_config_alias, plugin_config_path
    if str(reference).startswith("plugin:"):
        return plugin_config_path(str(reference), repository_root)
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
        base = root / section if section.startswith("experiments/") and (root / "experiments").is_dir() else root / "configs" / section
        candidate = (base / name).resolve()
        if candidate.is_relative_to(base.resolve()) and candidate.is_file():
            return candidate
    for candidate in names:
        alias = plugin_config_alias(f"{section}/{candidate.as_posix()}", root)
        if alias:
            return plugin_config_path(alias, root)
    return direct
