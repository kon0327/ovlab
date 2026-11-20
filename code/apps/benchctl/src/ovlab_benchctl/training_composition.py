"""Strict bounded ``extends`` composition shared by training entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os

from .config_references import ConfigReferences

from .strict_yaml import load
from .training_errors import TrainingProfileError


def merge_documents(parent: dict, child: dict) -> dict:
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        result[key] = merge_documents(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_training_document(path: str | Path, *, config_root: Path | None = None, stack=(), expected_kind=None, repository_root=None) -> dict:
    """Compose local training overrides with registered project defaults.

    Relative parents remain bounded to their owner; ``plugin://`` explicitly
    selects another project's configuration tree.
    """
    raw_path = str(path)
    candidate = Path(path).expanduser() if not raw_path.startswith(("plugin:", "ovlab:")) else None
    if repository_root is None:
        repository_root = os.environ.get("OVLAB_ROOT")
    if repository_root is None and candidate is not None:
        absolute = candidate.resolve()
        repository_root = next((parent for parent in absolute.parents if (parent / "code").is_dir() and ((parent / "configs").is_dir() or (parent / "config").is_dir())), None)
        if repository_root is None:
            owner = Path(config_root).resolve() if config_root else next((p for p in absolute.parents if p.name in {"configs", "config", "experiments", "examples"}), absolute.parent)
            repository_root = owner.parent
    refs = ConfigReferences(repository_root or Path.cwd(), config_root=config_root)
    path = refs.source(raw_path) if raw_path.startswith(("plugin:", "ovlab:")) else candidate.resolve()
    root = Path(config_root).resolve() if config_root else None
    if path in stack:
        raise TrainingProfileError("training extends cycle: " + " -> ".join(str(item) for item in (*stack, path)))
    if stack and root is not None and not path.is_relative_to(root):
        raise TrainingProfileError(f"training extends escapes configuration root: {path}")
    raw = load(path)
    if expected_kind is not None and raw.get("kind") != expected_kind:
        raise TrainingProfileError(f"training profile kind must be {expected_kind}: {path}")
    reference = raw.get("extends")
    if reference is None:
        return raw
    if config_root is None:
        owner = next((p for p in path.parents if p.name in {"configs", "config", "experiments", "examples"}), path.parent)
        refs = ConfigReferences(refs.repository_root, config_root=owner)
    try:
        parent_path = refs.parent(path, reference)
    except ValueError as exc:
        raise TrainingProfileError(str(exc)) from exc
    parent_root = refs.origin(parent_path)
    parent = load_training_document(parent_path, config_root=parent_root, stack=(*stack, path), expected_kind=expected_kind, repository_root=refs.repository_root)
    if "schema_version" in raw and raw["schema_version"] != parent.get("schema_version"):
        raise TrainingProfileError("training extends must preserve its parent's schema_version")
    return merge_documents(parent, raw)


def load_training_profile(path: str | Path, *, config_root: Path | None = None):
    from .training_profiles import TRAINING_PROFILE_KIND, TrainingProfile
    return TrainingProfile.from_document(load_training_document(path, config_root=config_root, expected_kind=TRAINING_PROFILE_KIND))
