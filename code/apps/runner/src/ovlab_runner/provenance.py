"""Injectable run provenance without implicit environment or Git mutation."""

from dataclasses import dataclass, field

from ovlab_core.contracts import Metadata, normalize_metadata


@dataclass(frozen=True, slots=True)
class ProvenanceSnapshot:
    ovlab_git_commit: str | None = None
    ovlab_dirty: bool | None = None
    external_commits: Metadata = field(default_factory=dict)
    environment_snapshot_reference: str | None = None
    checkpoint_identity: str | None = None
    dataset_identity: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "external_commits", normalize_metadata(self.external_commits, type(self).__name__))

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class StaticProvenanceProvider:
    def __init__(self, snapshot=None): self.snapshot = snapshot or ProvenanceSnapshot()
    def collect(self): return self.snapshot
