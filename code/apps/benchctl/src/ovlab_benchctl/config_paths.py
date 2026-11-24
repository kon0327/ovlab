"""Resolve explicit configuration origins or convenient user profile names."""
from __future__ import annotations
from pathlib import Path


def resolve_config_path(reference: str | Path, repository_root: Path, section: str) -> Path:
    from .config_references import ConfigReferences
    root = repository_root.expanduser().resolve()
    if str(reference).startswith(("plugin:", "module:", "ovlab:")):
        return ConfigReferences(root).source(str(reference))
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path.resolve()
    direct = (root / path).resolve()
    if direct.is_file() or (path.parts and path.parts[0] in {"configs", "config", "experiments", "examples"}):
        return direct
    names = [path] if path.suffix else [path, path.with_suffix(".yaml")]
    base = root / section if section.startswith(("experiments/", "examples/")) else root / "configs" / section
    bases = [base]
    if section.startswith("experiments/"):
        bases.append(root / "examples/experiments")
    candidates = set()
    for owner in bases:
        for name in names:
            candidate = (owner / name).resolve()
            if candidate.is_relative_to(owner.resolve()) and candidate.is_file():
                candidates.add(candidate)
    if len(candidates) > 1:
        from .errors import ConfigReferenceError
        raise ConfigReferenceError(f"ambiguous configuration name {reference!r}; select an explicit path")
    if candidates:
        return candidates.pop()
    return direct
