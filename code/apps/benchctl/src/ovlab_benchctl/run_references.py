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
