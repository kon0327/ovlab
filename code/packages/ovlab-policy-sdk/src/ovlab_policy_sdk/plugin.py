"""Public policy plugin API v2, independent of orchestration/model libraries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from ovlab_core.contracts import ActionSpec, BenchmarkContext, PolicyCapabilities
    from .adapter import PolicyAdapter


class PolicyPluginValidationError(ValueError):
    """A plugin-owned document violates its schema."""


class PolicyPluginCompatibilityError(ValueError):
    """Valid inputs or artifacts cannot be used together by this plugin."""


class PolicyPluginReferenceError(ValueError):
    """A required resource or implementation is missing."""


class PolicyPluginUnavailableError(RuntimeError):
    """A selected integration cannot provide its declared runtime service."""


class PolicySettings(Protocol):
    """Minimum settings contract consumed by the generic experiment resolver."""

    @property
    def record_raw_output(self) -> bool: ...


class PolicyPlugin(ABC):
    """Stateless integration hooks; model/episode state belongs to PolicyAdapter.

    Control hooks must not load weights or fetch resources. They can execute on
    the host during planning, before the model's runtime dependencies exist.
    """

    api_version = "2"

    @abstractmethod
    def validate_policy(self, doc: dict, path: str) -> None:
        """Validate the complete composed policy document."""

    def validate_checkpoint_template(self, doc: dict, path: str) -> None:
        """Validate a policy awaiting a workflow training checkpoint.

        Called only when the workflow explicitly binds ``checkpoint_from``.
        Plugins may omit the identity that their checkpoint-binding hook will
        supply, but must validate every other setting. Runtime validation always
        calls ``validate_policy`` on the fully bound document.
        """
        self.validate_policy(doc, path)

    @abstractmethod
    def resolve_settings(self, doc: dict, benchmark_context: BenchmarkContext | None, action_spec: ActionSpec,
                         checkpoints: Mapping, devices: Mapping, *,
                         repository_root: Path, environment: Mapping[str, str]) -> PolicySettings:
        """Resolve model settings from canonical peer contracts and resources.

        Standalone serving receives None; a model never reads benchmark YAML.
        """

    @abstractmethod
    def create_adapter(self, settings: PolicySettings) -> PolicyAdapter:
        """Return an adapter in CREATED state; load weights in its _initialize."""

    @abstractmethod
    def identity(self, capabilities: PolicyCapabilities) -> Mapping[str, object]:
        """Return RPC model/normalization/prompt/codec identity and runtime_versions.

        The optional method_descriptor records adaptation or method identity.
        Values must be serializable metadata; no runtime objects or paths that
        would require access to the registering host belong in this response.
        """

    def validate_document(self, doc: dict, path: str) -> None:
        """Optional validation for kinds declared in registry document_kinds."""
        raise PolicyPluginValidationError(f"unsupported plugin document: {doc.get('kind')}")

    def resolve_document(self, doc: dict, *, runtime: bool = False) -> Mapping:
        """Optional descriptor-only resolution; never imply readiness falsely."""
        raise PolicyPluginCompatibilityError("this plugin has no descriptor resolver")

    def describe_document(self, doc: dict) -> Mapping:
        return self.resolve_document(doc)

    @abstractmethod
    def describe_capabilities(self, settings: PolicySettings) -> PolicyCapabilities:
        """Declare static inputs/actions without constructing or loading a model."""

    def validate_observation(self, doc: dict, benchmark_context: BenchmarkContext) -> None:
        """Check named inputs, shapes and preprocessing compatibility."""

    def deployment_context(self, policy: dict, deployment: dict) -> None:
        """Standalone serving has no benchmark peer or scientific task identity."""
        return None

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
