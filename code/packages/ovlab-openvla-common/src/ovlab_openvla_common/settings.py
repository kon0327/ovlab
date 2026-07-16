"""Immutable model-source descriptors shared by OpenVLA policies."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenVlaModelSource:
    source: str
    revision: str | None = None
    expected_checksum: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("model source must not be empty")
        object.__setattr__(self, "source", self.source.strip())
        for name in ("revision", "expected_checksum"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
            if value is not None:
                object.__setattr__(self, name, value.strip())
