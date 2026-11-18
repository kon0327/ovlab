"""Public policy plugin API v1, independent of orchestration/model libraries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from ovlab_core.contracts import ActionSpec, PolicyCapabilities
    from .adapter import PolicyAdapter


class PolicyPluginValidationError(ValueError):
    """A plugin-owned document violates its schema."""


class PolicyPluginCompatibilityError(ValueError):
    """Valid inputs or artifacts cannot be used together by this plugin."""


class PolicyPluginReferenceError(ValueError):
    """A required resource or implementation is missing."""


class PolicySettings(Protocol):
    """Minimum settings contract consumed by the generic experiment resolver."""

    @property
    def record_raw_output(self) -> bool: ...


class PolicyPlugin(ABC):
    """Stateless integration hooks; model/episode state belongs to PolicyAdapter.

    Control hooks must not load weights or fetch resources. They can execute on
    the host during planning, before the model's runtime dependencies exist.
    """

    api_version = "1"

    @abstractmethod
    def validate_policy(self, doc: dict, path: str) -> None:
        """Validate the complete composed policy document."""

    @abstractmethod
    def resolve_settings(self, doc: dict, benchmark_doc: dict, action_spec: ActionSpec,
                         checkpoints: Mapping, devices: Mapping, *,
                         repository_root: Path, environment: Mapping[str, str]) -> PolicySettings:
        """Construct settings from validated documents and resolved resources."""

    @abstractmethod
    def create_adapter(self, settings: PolicySettings) -> PolicyAdapter:
        """Return an adapter in CREATED state; load weights in its _initialize."""

    @abstractmethod
    def identity(self, capabilities: PolicyCapabilities) -> Mapping[str, object]:
        """Return model_identity, method_identity and runtime_versions for RPC."""

    def validate_document(self, doc: dict, path: str) -> None:
        """Optional validation for kinds declared in registry document_kinds."""
        raise PolicyPluginValidationError(f"unsupported plugin document: {doc.get('kind')}")

    def resolve_document(self, doc: dict, *, runtime: bool = False) -> Mapping:
        """Optional descriptor-only resolution; never imply readiness falsely."""
        raise PolicyPluginCompatibilityError("this plugin has no descriptor resolver")

    def describe_document(self, doc: dict) -> Mapping:
        return self.resolve_document(doc)

    def validate_observation(self, doc: dict, benchmark_doc: dict) -> None:
        """Check named inputs, shapes and preprocessing compatibility."""

    def validate_checkpoint(self, entry: dict, path: str) -> None:
        """Validate plugin-specific artifact/method metadata beyond file hashes."""

    def scientific_policy(self, doc: dict) -> dict:
        """Return portable scientific settings, retaining immutable identities."""
        return deepcopy(doc)

    def runtime_resources(self, doc: dict, repository_root: Path, environment: Mapping) -> dict:
        """Return extra execution metadata without changing scientific identity."""
        return {}

    def runtime_mounts(self, doc: dict, repository_root: Path, environment: Mapping) -> list[dict]:
        """Return absolute source/target directories, optionally environment/required_files."""
        return []

    def accepts_settings(self, settings: object) -> bool:
        """Compatibility hook for older direct callers; normal startup selects by type."""
        return False

    def accepts_capabilities(self, capabilities: PolicyCapabilities) -> bool:
        """Compatibility hook for older direct identity-provider callers."""
        return False
