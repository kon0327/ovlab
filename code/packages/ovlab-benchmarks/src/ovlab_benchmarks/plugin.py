"""Dependency-free benchmark control interface; runtime contracts load on demand."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ovlab_core.contracts import BenchmarkCapabilities, BenchmarkIdentity, BenchmarkResolutionContext
    from .adapter import BenchmarkAdapter
    from .catalog import AssetRequirement, CapabilityRefinement, TaskCatalog, TaskSelection

class BenchmarkPluginValidationError(ValueError):
    """A plugin-owned document violates its schema."""


class BenchmarkPluginCompatibilityError(ValueError):
    """Valid inputs cannot satisfy the benchmark's declared contract."""


class BenchmarkPluginReferenceError(ValueError):
    """A selected resource or implementation is unavailable."""


class BenchmarkPluginUnavailableError(RuntimeError):
    """The selected benchmark runtime cannot be started."""


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkPluginValidationError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ConfigReference:
    reference: str
    kind: str

    def __post_init__(self) -> None:
        _text(self.reference, "reference")
        _text(self.kind, "kind")


@dataclass(frozen=True, slots=True)
class BenchmarkRuntimeMount:
    source: str
    target: str
    read_only: bool = True
    required_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.source, "mount source")
        _text(self.target, "mount target")
        target = PurePosixPath(self.target)
        if not target.is_absolute() or ".." in target.parts or str(target) == "/" or "\\" in self.target:
            raise BenchmarkPluginValidationError("mount target must be an absolute non-root container path")
        if type(self.read_only) is not bool:
            raise BenchmarkPluginValidationError("mount read_only must be a boolean")
        required = tuple(self.required_files)
        for value in required:
            _text(value, "required file")
            path = PurePosixPath(value)
            if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
                raise BenchmarkPluginValidationError("required files must remain inside the mount")
        object.__setattr__(self, "target", str(target))
        object.__setattr__(self, "required_files", required)


@dataclass(frozen=True, slots=True)
class BenchmarkDeploymentSpec:
    """Runtime requirements; the orchestrator retains ownership of Compose."""

    build_role: str
    image: str
    image_env: str
    requires_gpu: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[BenchmarkRuntimeMount, ...] = ()

    def __post_init__(self) -> None:
        for name in ("build_role", "image", "image_env"):
            _text(getattr(self, name), name)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.image_env) is None:
            raise BenchmarkPluginValidationError("image_env must be an environment variable name")
        if type(self.requires_gpu) is not bool:
            raise BenchmarkPluginValidationError("requires_gpu must be a boolean")
        if not isinstance(self.environment, Mapping) or any(
            not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or not isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise BenchmarkPluginValidationError("environment must be a string mapping")
        mounts = tuple(self.mounts)
        if any(not isinstance(mount, BenchmarkRuntimeMount) for mount in mounts):
            raise BenchmarkPluginValidationError("mounts must contain BenchmarkRuntimeMount values")
        targets = [mount.target for mount in mounts]
        if len(set(targets)) != len(targets):
            raise BenchmarkPluginValidationError("mount targets must be unique")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "mounts", mounts)


class BenchmarkPluginSettings(Protocol):
    """Marker for immutable, plugin-owned settings; no policy action input."""


class BenchmarkPlugin(ABC):
    """Stateless control hooks; simulator and episode state belong to adapters.

    Control hooks must not import simulator runtimes, acquire assets, mutate the
    process environment, or start rendering. Runtime factories execute only in
    the selected worker and return an adapter in CREATED state.
    """

    api_version = "1"

    @abstractmethod
    def validate_benchmark(self, document: dict, path: str) -> None: ...

    @abstractmethod
    def resolve_settings(self, document: dict, context: BenchmarkResolutionContext) -> BenchmarkPluginSettings: ...

    @abstractmethod
    def identity(self, settings: BenchmarkPluginSettings) -> BenchmarkIdentity: ...

    @abstractmethod
    def describe_capabilities(self, settings: BenchmarkPluginSettings) -> BenchmarkCapabilities: ...

    @abstractmethod
    def task_catalog(self, settings: BenchmarkPluginSettings) -> TaskCatalog: ...

    def select_tasks(self, settings: BenchmarkPluginSettings, catalog: TaskCatalog) -> TaskSelection:
        return catalog.select()

    @abstractmethod
    def create_adapter(self, settings: BenchmarkPluginSettings) -> BenchmarkAdapter: ...

    def validate_execution_profile(self, document: dict, path: str) -> None:
        raise BenchmarkPluginValidationError(f"{path}: execution profiles are unsupported")

    def validate_local_execution(self, overrides: dict, path: str) -> None:
        if overrides:
            raise BenchmarkPluginValidationError(f"{path}: local execution overrides are unsupported")

    def execution_metadata(self, settings: BenchmarkPluginSettings) -> dict:
        return {}

    def method_claims(self, settings: BenchmarkPluginSettings) -> dict:
        return {}

    def capability_refinement(self, settings: BenchmarkPluginSettings) -> CapabilityRefinement:
        from .catalog import CapabilityRefinement
        return CapabilityRefinement()

    def asset_requirements(self, document: dict, settings: BenchmarkPluginSettings) -> tuple[AssetRequirement, ...]:
        return ()

    def bundle_references(self, document: dict, execution_profile: dict | None = None) -> tuple[ConfigReference, ...]:
        return ()

    def deployment_spec(self, settings: BenchmarkPluginSettings,
                        context: BenchmarkResolutionContext) -> BenchmarkDeploymentSpec:
        raise BenchmarkPluginUnavailableError("this benchmark has no container deployment")

    def scientific_benchmark(self, document: dict) -> dict:
        from copy import deepcopy
        return deepcopy(document)

    def resolve_deployment(self, document: dict, *, execution_profile: dict, local_execution: dict,
                           devices: Mapping, environment: Mapping, deployment: dict) -> BenchmarkDeploymentSpec:
        """Describe container requirements using only stdlib control dependencies."""
        raise BenchmarkPluginUnavailableError("this benchmark has no container deployment control hook")

def __getattr__(name):
    if name in {"AssetRequirement", "CapabilityRefinement", "TaskCatalog", "TaskSelection",
                "validate_runtime_capabilities", "create_benchmark_adapter"}:
        from . import catalog
        return getattr(catalog, name)
    raise AttributeError(name)
