"""Unambiguous short-hash references for benchmark and training runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


RUN_HASH_LENGTH = 8
_CANONICAL_SUFFIX = re.compile(r"_([0-9a-fA-F]{8})$")
_RUN_HASH = re.compile(r"^[0-9a-fA-F]{8}$")


class RunReferenceError(RuntimeError):
    """A run reference cannot be resolved safely."""


class RunReferenceUnavailableError(RunReferenceError):
    """No run matches a supplied ID or short hash."""


class RunReferenceAmbiguousError(RunReferenceError):
    """More than one run matches a supplied short hash."""


@dataclass(frozen=True, slots=True)
class ResolvedRunReference:
    run_id: str
    run_hash: str
    path: Path


def run_hash(run_id: str) -> str:
    """Return the displayed short lookup hash for a run ID.

    Current human-readable IDs already end in their canonical eight-character
    hash. Legacy IDs receive a stable compatibility alias derived from the ID;
    this alias is a locator, not a scientific or artifact-integrity hash.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run ID must be a non-empty string")
    match = _CANONICAL_SUFFIX.search(run_id)
    if match is not None:
        return match.group(1).lower()
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:RUN_HASH_LENGTH]


def resolve_run_reference(
    root: str | Path,
    reference: str,
    *,
    label: str = "run",
) -> ResolvedRunReference:
    """Resolve one exact directory ID or one displayed short hash.

    Exact IDs win. Hash lookup requires an exact eight-character hash and fails
    closed when aliases collide.
    """
    root_path = Path(root).expanduser().resolve()
    if not isinstance(reference, str) or not reference:
        raise RunReferenceUnavailableError(f"{label} reference must be a non-empty string")
    candidate = Path(reference)
    if len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise RunReferenceUnavailableError(
            f"{label} reference must be one run ID or an {RUN_HASH_LENGTH}-character hash"
        )

    exact = root_path / reference
    if exact.is_symlink():
        raise RunReferenceError(f"{label} reference resolves to a forbidden symbolic link: {reference}")
    if exact.is_dir():
        return ResolvedRunReference(reference, run_hash(reference), exact.resolve())

    if _RUN_HASH.fullmatch(reference) is None:
        raise RunReferenceUnavailableError(
            f"{label} is unavailable: {reference!r}; use its full ID or displayed "
            f"{RUN_HASH_LENGTH}-character hash"
        )
    requested_hash = reference.lower()
    matches: list[ResolvedRunReference] = []
    if root_path.is_dir():
        for path in sorted(root_path.iterdir()):
            if path.is_symlink() or not path.is_dir():
                continue
            identifier = path.name
            if run_hash(identifier) == requested_hash:
                matches.append(ResolvedRunReference(identifier, requested_hash, path.resolve()))
    if not matches:
        raise RunReferenceUnavailableError(
            f"no {label} matches hash {requested_hash!r} under {root_path}"
        )
    if len(matches) > 1:
        identifiers = ", ".join(item.run_id for item in matches)
        raise RunReferenceAmbiguousError(
            f"{label} hash {requested_hash!r} is ambiguous; matching IDs: {identifiers}"
        )
    return matches[0]


def resolve_run_reference_across(
    roots: tuple[str | Path, ...], reference: str, *, label: str = "run",
) -> ResolvedRunReference:
    """Resolve one run across typed roots, failing closed on hash collisions."""
    root_paths = tuple(dict.fromkeys(Path(root).expanduser().resolve() for root in roots))
    if not isinstance(reference, str) or not reference:
        raise RunReferenceUnavailableError(f"{label} reference must be a non-empty string")
    candidate = Path(reference)
    if len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise RunReferenceUnavailableError(
            f"{label} reference must be one run ID or an {RUN_HASH_LENGTH}-character hash"
        )
    exact = []
    for root in root_paths:
        path = root / reference
        if path.is_symlink():
            raise RunReferenceError(f"{label} reference resolves to a forbidden symbolic link: {reference}")
        if path.is_dir():
            exact.append(ResolvedRunReference(reference, run_hash(reference), path.resolve()))
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RunReferenceAmbiguousError(f"{label} ID exists in multiple run categories: {reference}")
    if _RUN_HASH.fullmatch(reference) is None:
        raise RunReferenceUnavailableError(
            f"{label} is unavailable: {reference!r}; use its full ID or displayed "
            f"{RUN_HASH_LENGTH}-character hash"
        )
    requested_hash = reference.lower()
    matches = []
    for root in root_paths:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_symlink() or not path.is_dir():
                continue
            if run_hash(path.name) == requested_hash:
                matches.append(ResolvedRunReference(path.name, requested_hash, path.resolve()))
    unique = {item.path: item for item in matches}
    if not unique:
        raise RunReferenceUnavailableError(f"no {label} matches hash {requested_hash!r} under configured run roots")
    if len(unique) > 1:
        identifiers = ", ".join(item.run_id for item in unique.values())
        raise RunReferenceAmbiguousError(f"{label} hash {requested_hash!r} is ambiguous; matching IDs: {identifiers}")
    return next(iter(unique.values()))


def resolve_run_reference_tree(
    runs_root: str | Path,
    reference: str,
    *,
    category: str | None = None,
    label: str = "run",
) -> ResolvedRunReference:
    """Find an immutable run below legacy or experiment-scoped run roots.

    A bare ID or short hash must identify exactly one run. Callers can restrict
    the search to train/evaluate, but must not silently pick one experiment when
    the same human ID exists in several experiment trees.
    """
    if category not in {None, "train", "evaluate"}:
        raise ValueError("category must be train or evaluate")
    if not isinstance(reference, str) or not reference:
        raise RunReferenceUnavailableError(f"{label} reference must be a run ID, hash, or absolute canonical run path")
    root = Path(runs_root).expanduser().resolve()
    if not root.is_dir():
        raise RunReferenceUnavailableError(f"{label} is unavailable: {reference!r}")

    def path_category(path: Path) -> str | None:
        parts = path.relative_to(root).parts
        if not parts:
            return None
        if parts[0] in {"train", "evaluate"}:
            return parts[0]
        if len(parts) >= 3 and parts[1] in {"train", "evaluate"}:
            return parts[1]
        return "evaluate" if len(parts) == 1 else None

    requested_path = Path(reference).expanduser()
    if requested_path.is_absolute():
        if ".." in requested_path.parts or not requested_path.is_relative_to(root):
            raise RunReferenceUnavailableError(f"{label} path is outside the configured runs root: {reference}")
        current = root
        for part in requested_path.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise RunReferenceError(f"{label} path contains a forbidden symbolic link: {reference}")
        resolved = requested_path.resolve()
        if not resolved.is_relative_to(root):
            raise RunReferenceUnavailableError(f"{label} path is outside the configured runs root: {reference}")
        if (not resolved.is_dir() or
                not any((resolved / marker).is_file() for marker in (
                    "manifest.started.json", "manifest.completed.json", "manifest.failed.json", "training-plan.json",
                )) or (category is not None and path_category(resolved) != category)):
            raise RunReferenceUnavailableError(f"{label} path is not a canonical {category or ''} run: {reference}")
        return ResolvedRunReference(resolved.name, run_hash(resolved.name), resolved)
    if requested_path.parts != (reference,) or reference in {".", ".."}:
        raise RunReferenceUnavailableError(f"{label} reference must be one run ID or an {RUN_HASH_LENGTH}-character hash")

    matches: list[ResolvedRunReference] = []
    pending = [root]
    while pending:
        parent = pending.pop()
        for path in sorted(parent.iterdir(), reverse=True):
            if path.is_symlink():
                if path.name == reference:
                    raise RunReferenceError(f"{label} reference resolves to a forbidden symbolic link: {reference}")
                continue
            if not path.is_dir():
                continue
            if any((path / marker).is_file() for marker in (
                "manifest.started.json", "manifest.completed.json", "manifest.failed.json", "training-plan.json",
            )):
                if category is None or path_category(path) == category:
                    matches.append(ResolvedRunReference(path.name, run_hash(path.name), path.resolve()))
                continue
            pending.append(path)
    exact = [item for item in matches if item.run_id == reference]
    selected = exact if exact else [item for item in matches if _RUN_HASH.fullmatch(reference) and item.run_hash == reference.lower()]
    if not selected:
        raise RunReferenceUnavailableError(f"{label} is unavailable: {reference!r}")
    if len(selected) > 1:
        locations = ", ".join(str(item.path.relative_to(root)) for item in selected)
        raise RunReferenceAmbiguousError(f"{label} reference {reference!r} is ambiguous: {locations}")
    return selected[0]
