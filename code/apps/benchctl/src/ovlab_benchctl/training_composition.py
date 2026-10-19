"""Strict bounded ``extends`` composition shared by training entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .strict_yaml import load
from .training_errors import TrainingProfileError


def merge_documents(parent: dict, child: dict) -> dict:
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        result[key] = merge_documents(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_training_document(path: str | Path, *, config_root: Path | None = None, stack=()) -> dict:
    path = Path(path).expanduser().resolve()
    if config_root is None:
        config_root = next((parent for parent in path.parents if parent.name == "configs"), path.parent)
    root = Path(config_root).resolve()
    if path in stack:
        raise TrainingProfileError("training extends cycle: " + " -> ".join(str(item) for item in (*stack, path)))
    if stack and not path.is_relative_to(root):
        raise TrainingProfileError(f"training extends escapes configuration root: {path}")
    raw = load(path)
    reference = raw.get("extends")
    if reference is None:
        return raw
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise TrainingProfileError("training extends must be a nonempty relative YAML path")
    parent_path = (path.parent / reference).resolve()
    if not parent_path.is_relative_to(root):
        raise TrainingProfileError(f"training extends escapes configuration root: {parent_path}")
    parent = load_training_document(parent_path, config_root=root, stack=(*stack, path))
    if "schema_version" in raw and raw["schema_version"] != parent.get("schema_version"):
        raise TrainingProfileError("training extends must preserve its parent's schema_version")
    return merge_documents(parent, raw)


def load_training_profile(path: str | Path, *, config_root: Path | None = None):
    from .training_profiles import TrainingProfile
    return TrainingProfile.from_document(load_training_document(path, config_root=config_root))
