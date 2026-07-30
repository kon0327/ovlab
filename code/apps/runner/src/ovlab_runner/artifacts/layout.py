"""Deterministic filesystem-safe artifact keys."""

import hashlib
import re

from ..errors import ArtifactError


_READABLE_RUN_KEY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9a-f]{8}$"
)


def safe_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ArtifactError("artifact identifiers must be non-empty and contain no NUL")
    if any(part == ".." for part in value.replace("\\", "/").split("/")):
        raise ArtifactError("artifact identifiers must not contain path traversal")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")[:48] or "id"
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def run_key(value: str) -> str:
    """Preserve canonical readable run names while retaining legacy mapping."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ArtifactError("artifact identifiers must be non-empty and contain no NUL")
    if len(value.encode("utf-8")) <= 240 and _READABLE_RUN_KEY.fullmatch(value):
        return value
    return safe_key(value)
