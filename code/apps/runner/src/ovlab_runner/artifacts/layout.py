"""Deterministic filesystem-safe artifact keys."""

import hashlib
import re

from ..errors import ArtifactError


def safe_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ArtifactError("artifact identifiers must be non-empty and contain no NUL")
    if any(part == ".." for part in value.replace("\\", "/").split("/")):
        raise ArtifactError("artifact identifiers must not contain path traversal")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")[:48] or "id"
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"
