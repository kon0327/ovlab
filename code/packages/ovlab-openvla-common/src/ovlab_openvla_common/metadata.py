"""Reproducible checkpoint identity contracts."""

from dataclasses import dataclass, field
from ovlab_core.contracts import Metadata, normalize_metadata


@dataclass(frozen=True, slots=True)
class OpenVlaCheckpointIdentity:
    configured_source: str
    resolved_local_path: str | None
    openvla_git_commit: str
    model_identity: str
    processor_identity: str
    unnorm_key: str
    action_statistics_identity: str
    snapshot_revision: str | None
    expected_checksum: str | None
    settings_hash: str
    identity_strength: str = "revision-metadata"
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = ("configured_source", "openvla_git_commit", "model_identity", "processor_identity",
                    "unnorm_key", "action_statistics_identity", "settings_hash", "identity_strength")
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name).strip() for name in required):
            raise ValueError("checkpoint identity required fields must be non-empty strings")
        for name in ("resolved_local_path", "snapshot_revision", "expected_checksum"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))

    def as_metadata(self) -> dict[str, object]:
        return {
            "configured_source": self.configured_source,
            "resolved_local_path": self.resolved_local_path,
            "openvla_git_commit": self.openvla_git_commit,
            "model_identity": self.model_identity,
            "processor_identity": self.processor_identity,
            "unnorm_key": self.unnorm_key,
            "action_statistics_identity": self.action_statistics_identity,
            "snapshot_revision": self.snapshot_revision,
            "expected_checksum": self.expected_checksum,
            "settings_hash": self.settings_hash,
            "identity_strength": self.identity_strength,
        }
