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


from .plugin import (BenchmarkPlugin, BenchmarkPluginSettings, BenchmarkPluginValidationError,
    BenchmarkPluginCompatibilityError, BenchmarkPluginReferenceError, _text)

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
