"""Immutable benchmark descriptions shared with policy control integrations.

These values contain no simulator objects, orchestration plans, or plugin-owned
configuration documents. A policy can inspect a peer without depending on the
benchmark SDK or importing the peer's implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .capabilities import BenchmarkCapabilities
from .errors import validation_error
from .identifiers import TaskId
from .metadata import Metadata, normalize_metadata


def _text(value: object, contract: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(contract, name, "must be a non-empty string")


def _strings(values: Mapping[str, str], contract: str, name: str) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise validation_error(contract, name, "must be a string mapping")
    result = {}
    for key, value in values.items():
        _text(key, contract, name)
        if not isinstance(value, str):
            raise validation_error(contract, f"{name}.{key}", "must be a string")
        result[key] = value
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    plugin_id: str
    benchmark_id: str
    version: str
    source_revision: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for name in ("plugin_id", "benchmark_id", "version", "source_revision"):
            _text(getattr(self, name), contract, name)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class BenchmarkTaskSetIdentity:
    catalog_id: str
    catalog_sha256: str
    suite_ids: tuple[str, ...]
    task_ids: tuple[TaskId, ...]

    def __post_init__(self) -> None:
        contract = type(self).__name__
        _text(self.catalog_id, contract, "catalog_id")
        digest = self.catalog_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise validation_error(contract, "catalog_sha256", "must be a lowercase SHA-256 digest")
        suites, tasks = tuple(self.suite_ids), tuple(self.task_ids)
        for suite in suites:
            _text(suite, contract, "suite_ids")
        if not suites or len(set(suites)) != len(suites):
            raise validation_error(contract, "suite_ids", "must be non-empty and unique")
        if not tasks or any(not isinstance(task, TaskId) for task in tasks) or len(set(tasks)) != len(tasks):
            raise validation_error(contract, "task_ids", "must contain non-empty, unique TaskId values")
        object.__setattr__(self, "suite_ids", suites)
        object.__setattr__(self, "task_ids", tasks)


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    identity: BenchmarkIdentity
    capabilities: BenchmarkCapabilities
    task_set: BenchmarkTaskSetIdentity
    method_claims: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for name, expected in (("identity", BenchmarkIdentity), ("capabilities", BenchmarkCapabilities),
                               ("task_set", BenchmarkTaskSetIdentity)):
            if not isinstance(getattr(self, name), expected):
                raise validation_error(contract, name, f"must be {expected.__name__}")
        if not set(self.task_set.suite_ids) <= set(self.capabilities.task_suites):
            raise validation_error(contract, "task_set", "selected suites must be declared in capabilities")
        object.__setattr__(self, "method_claims", normalize_metadata(self.method_claims, contract, "method_claims"))


@dataclass(frozen=True, slots=True)
class BenchmarkResolutionContext:
    """Already resolved control inputs; constructing this value performs no IO."""

    base_seed: int
    maximum_episode_steps: int
    repository_root: Path | None = None
    devices: Mapping[str, str] = field(default_factory=dict)
    paths: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    execution: Metadata = field(default_factory=dict)
    assets: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise validation_error(contract, "base_seed", "must be a non-negative integer")
        if type(self.maximum_episode_steps) is not int or self.maximum_episode_steps <= 0:
            raise validation_error(contract, "maximum_episode_steps", "must be a positive integer")
        if self.repository_root is not None and not isinstance(self.repository_root, Path):
            raise validation_error(contract, "repository_root", "must be a Path or None")
        for name in ("devices", "paths", "environment"):
            object.__setattr__(self, name, _strings(getattr(self, name), contract, name))
        for name in ("execution", "assets"):
            object.__setattr__(self, name, normalize_metadata(getattr(self, name), contract, name))
