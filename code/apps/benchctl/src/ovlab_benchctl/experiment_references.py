"""Shared, unambiguous references to complete experiment instances."""

from pathlib import Path
import re

from .run_references import RunReferenceAmbiguousError, RunReferenceError


def find_experiment(runs_root: Path, reference: str) -> Path | None:
    """Resolve a direct instance path/ID or its displayed suffix hash.

    Modern instance IDs end in ten hex characters; migrated instances can end
    in an eight-character run hash. Never choose silently between collisions.
    """
    root = runs_root.expanduser().resolve()
    requested = Path(reference).expanduser()
    if requested.is_absolute():
        if ".." in requested.parts or requested.parent != root:
            return None
        candidates = [requested]
    elif requested.parts == (reference,) and reference not in {".", ".."}:
        direct = root / reference
        if direct.is_symlink():
            raise RunReferenceError(f"experiment reference is a symbolic link: {reference}")
        if direct.is_dir():
            candidates = [direct]
        elif re.fullmatch(r"(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{10})", reference) and root.is_dir():
            candidates = [p for p in root.iterdir() if p.name.lower().endswith("_" + reference.lower())]
        else:
            return None
    else:
        return None
    found = []
    for directory in candidates:
        if directory.is_symlink():
            raise RunReferenceError(f"experiment reference is a symbolic link: {directory}")
        metadata = directory / ".experiment"
        if metadata.is_symlink():
            raise RunReferenceError(f"experiment metadata is a symbolic link: {metadata}")
        if directory.is_dir() and metadata.is_dir():
            found.append(directory)
    if len(found) > 1:
        raise RunReferenceAmbiguousError(
            f"experiment reference {reference!r} is ambiguous: " + ", ".join(p.name for p in sorted(found))
        )
    return found[0] if found else None
