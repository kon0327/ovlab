"""Control-plane benchmark plugin API, independent of simulator runtimes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ovlab_core.contracts import (
    AdapterState, BenchmarkCapabilities, BenchmarkIdentity, BenchmarkResolutionContext,
    BenchmarkTaskSetIdentity, Metadata, SignalAccess, SignalSpec, TaskId, normalize_metadata,
)

from .adapter import BenchmarkAdapter
from .contracts import TaskDescriptor


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


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TaskSelection:
    identity: BenchmarkTaskSetIdentity
    tasks: tuple[TaskDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BenchmarkTaskSetIdentity):
            raise BenchmarkPluginValidationError("selection identity must be BenchmarkTaskSetIdentity")
        tasks = tuple(self.tasks)
        if any(not isinstance(task, TaskDescriptor) for task in tasks):
            raise BenchmarkPluginValidationError("selection must contain TaskDescriptor values")
        if tuple(task.task_id for task in tasks) != self.identity.task_ids:
            raise BenchmarkPluginValidationError("selection tasks must match identity task IDs in order")
        if tuple(dict.fromkeys(task.suite_name for task in tasks)) != self.identity.suite_ids:
            raise BenchmarkPluginValidationError("selection suites must match identity suites in order")
        object.__setattr__(self, "tasks", tasks)

    @property
    def task_ids(self) -> tuple[TaskId, ...]:
        return self.identity.task_ids


@dataclass(frozen=True, slots=True)
class TaskCatalog:
    """Portable authoritative tasks, with no simulator handles or initial states."""

    catalog_id: str
    revision: str
    tasks: tuple[TaskDescriptor, ...]

    def __post_init__(self) -> None:
        _text(self.catalog_id, "catalog_id")
        _text(self.revision, "revision")
        tasks = tuple(self.tasks)
        if not tasks or any(not isinstance(task, TaskDescriptor) for task in tasks):
            raise BenchmarkPluginValidationError("catalog must contain TaskDescriptor values")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise BenchmarkPluginValidationError("catalog task IDs must be unique")
        if len({(task.suite_name, task.task_index) for task in tasks}) != len(tasks):
            raise BenchmarkPluginValidationError("catalog task indices must be unique within a suite")
        object.__setattr__(self, "tasks", tasks)

    def canonical(self) -> dict:
        return {"catalog_id": self.catalog_id, "revision": self.revision, "tasks": _plain(self.tasks)}

    @property
    def sha256(self) -> str:
        value = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def select(self, task_ids: tuple[TaskId, ...] | None = None, *,
               suite_names: tuple[str, ...] | None = None) -> TaskSelection:
        """Preserve explicit ID order, or authoritative catalog order for suites."""
        if task_ids is not None and suite_names is not None:
            raise BenchmarkPluginValidationError("select task IDs or suites, not both")
        if task_ids is not None:
            selected_ids = tuple(task_ids)
            if not selected_ids or any(not isinstance(item, TaskId) for item in selected_ids):
                raise BenchmarkPluginValidationError("selected task IDs must be non-empty TaskId values")
            if len(set(selected_ids)) != len(selected_ids):
                raise BenchmarkPluginValidationError("selected task IDs must be unique")
            by_id = {task.task_id: task for task in self.tasks}
            missing = [str(item) for item in selected_ids if item not in by_id]
            if missing:
                raise BenchmarkPluginReferenceError(f"unknown selected task IDs: {', '.join(missing)}")
            selected = tuple(by_id[item] for item in selected_ids)
        elif suite_names is not None:
            suites = tuple(suite_names)
            if not suites or any(not isinstance(item, str) or not item.strip() for item in suites):
                raise BenchmarkPluginValidationError("selected suites must be non-empty strings")
            if len(set(suites)) != len(suites):
                raise BenchmarkPluginValidationError("selected suites must be unique")
            missing = sorted(set(suites) - {task.suite_name for task in self.tasks})
            if missing:
                raise BenchmarkPluginReferenceError(f"unknown selected suites: {', '.join(missing)}")
            selected = tuple(task for task in self.tasks if task.suite_name in suites)
        else:
            selected = self.tasks
        identity = BenchmarkTaskSetIdentity(
            self.catalog_id, self.sha256, tuple(dict.fromkeys(task.suite_name for task in selected)),
            tuple(task.task_id for task in selected),
        )
        return TaskSelection(identity, selected)


@dataclass(frozen=True, slots=True)
class ConfigReference:
    reference: str
    kind: str

    def __post_init__(self) -> None:
        _text(self.reference, "reference")
        _text(self.kind, "kind")


@dataclass(frozen=True, slots=True)
class AssetRequirement:
    asset_id: str
    provider: str
    name: str
    version: str = "1"
    required: bool = True
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("asset_id", "provider", "name", "version"):
            _text(getattr(self, name), name)
        if type(self.required) is not bool:
            raise BenchmarkPluginValidationError("asset required must be a boolean")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))


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


@dataclass(frozen=True, slots=True)
class CapabilityRefinement:
    """Only these exact optional evaluation signals may be added at runtime."""

    optional_signals: tuple[SignalSpec, ...] = ()

    def __post_init__(self) -> None:
        signals = tuple(self.optional_signals)
        if any(not isinstance(item, SignalSpec) or not item.optional
               or item.access is SignalAccess.POLICY_VISIBLE for item in signals):
            raise BenchmarkPluginValidationError("refinements must be optional non-policy SignalSpec values")
        if len({item.name for item in signals}) != len(signals):
            raise BenchmarkPluginValidationError("refinement signal names must be unique")
        object.__setattr__(self, "optional_signals", signals)


def validate_runtime_capabilities(declared: BenchmarkCapabilities, actual: BenchmarkCapabilities,
                                  refinement: CapabilityRefinement | None = None) -> None:
    """Reject changed peer contracts and undeclared runtime signal extensions."""
    if not isinstance(declared, BenchmarkCapabilities) or not isinstance(actual, BenchmarkCapabilities):
        raise BenchmarkPluginValidationError("capabilities must be BenchmarkCapabilities values")
    refinement = refinement or CapabilityRefinement()
    if not isinstance(refinement, CapabilityRefinement):
        raise BenchmarkPluginValidationError("refinement must be CapabilityRefinement")
    for item in fields(declared):
        if item.name != "signal_registry" and _plain(getattr(declared, item.name)) != _plain(getattr(actual, item.name)):
            raise BenchmarkPluginCompatibilityError(f"runtime benchmark capability differs: {item.name}")
    promised = {spec.name: spec for spec in declared.signal_registry}
    observed = {spec.name: spec for spec in actual.signal_registry}
    additions = {spec.name: spec for spec in refinement.optional_signals}
    if set(promised) & set(additions):
        raise BenchmarkPluginValidationError("refinements cannot replace declared signals")
    for name, spec in promised.items():
        if observed.get(name) != spec:
            raise BenchmarkPluginCompatibilityError(f"runtime benchmark signal differs: {name}")
    for name in set(observed) - set(promised):
        if observed[name] != additions.get(name):
            raise BenchmarkPluginCompatibilityError(f"undeclared runtime benchmark signal: {name}")


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

    def method_claims(self, settings: BenchmarkPluginSettings) -> Metadata:
        return normalize_metadata({}, type(self).__name__)

    def capability_refinement(self, settings: BenchmarkPluginSettings) -> CapabilityRefinement:
        return CapabilityRefinement()


def create_benchmark_adapter(plugin: BenchmarkPlugin, settings: BenchmarkPluginSettings) -> BenchmarkAdapter:
    """Worker factory guard shared by all integrations."""
    adapter = plugin.create_adapter(settings)
    if not isinstance(adapter, BenchmarkAdapter):
        raise BenchmarkPluginCompatibilityError("benchmark factory must return a BenchmarkAdapter in CREATED state")
    if adapter.state is not AdapterState.CREATED:
        try:
            adapter.close()
        finally:
            raise BenchmarkPluginCompatibilityError("benchmark factory must return a BenchmarkAdapter in CREATED state")
    return adapter
