"""Filesystem permissions for host-managed finalized artifacts."""

from __future__ import annotations

from pathlib import Path

from .errors import ArtifactError


MANAGED_DIRECTORY_MODE = 0o2770
MANAGED_FILE_MODE = 0o640


def finalize_managed_tree(root: str | Path) -> None:
    """Make a finalized tree group-readable and safely deletable by its host group."""
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ArtifactError(f"managed artifact root is not a safe directory: {root_path}")
    entries = [root_path, *sorted(root_path.rglob("*"))]
    symbolic_links = [path for path in entries if path.is_symlink()]
    if symbolic_links:
        raise ArtifactError(f"managed artifact contains a forbidden symbolic link: {symbolic_links[0]}")
    for path in entries:
        if path.is_dir():
            path.chmod(MANAGED_DIRECTORY_MODE)
        elif path.is_file():
            path.chmod(MANAGED_FILE_MODE)


def finalize_managed_directory(path: str | Path) -> None:
    """Apply the managed directory mode without traversing sibling artifacts."""
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactError(f"managed artifact parent is not a safe directory: {directory}")
    directory.chmod(MANAGED_DIRECTORY_MODE)
