"""Shared adapter states, observation specifications, and capabilities."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .actions import ActionSpec
from .errors import validation_error
from .metadata import Metadata, normalize_metadata
from .observation import ColorSpace, ImageEncoding
from .signals import SignalRegistry


class AdapterState(str, Enum):
    CREATED = "created"
    READY = "ready"
    EPISODE_ACTIVE = "episode_active"
    CLOSED = "closed"


def _non_empty(value: str, contract: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(contract, field_name, "must not be empty or whitespace-only")


def _count(value: int, contract: str, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise validation_error(contract, field_name, "must be a non-negative integer")


def _normalize_shapes(value: tuple[tuple[int, ...], ...], contract: str) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(tuple(shape) for shape in value)
    if not shapes:
        raise validation_error(contract, "shapes", "must contain at least one permitted shape")
    for shape in shapes:
        if not shape or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
            raise validation_error(contract, "shapes", "every shape must contain positive integer dimensions")
    if len(set(shapes)) != len(shapes):
        raise validation_error(contract, "shapes", "must not contain duplicates")
    return tuple(sorted(shapes))


def _normalize_dtype(value: str, contract: str) -> str:
    _non_empty(value, contract, "dtype")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise validation_error(contract, "dtype", "must name a NumPy dtype") from exc
    if dtype.hasobject:
        raise validation_error(contract, "dtype", "object dtype is not supported")
    return dtype.name


@dataclass(frozen=True, slots=True)
class ImageObservationSpec:
    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtype: str
    encodings: tuple[ImageEncoding, ...]
    color_spaces: tuple[ColorSpace, ...]
    required: bool = True
    minimum_count: int = 1
    maximum_count: int = 1
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        _non_empty(self.name, contract, "name")
        shapes = _normalize_shapes(self.shapes, contract)
        dtype = _normalize_dtype(self.dtype, contract)
        encodings = tuple(self.encodings)
        colors = tuple(self.color_spaces)
        if not encodings or any(not isinstance(value, ImageEncoding) for value in encodings):
            raise validation_error(contract, "encodings", "must contain ImageEncoding values")
        if not colors or any(not isinstance(value, ColorSpace) for value in colors):
            raise validation_error(contract, "color_spaces", "must contain ColorSpace values")
        _count(self.minimum_count, contract, "minimum_count")
        _count(self.maximum_count, contract, "maximum_count")
        if self.minimum_count > self.maximum_count:
            raise validation_error(contract, "minimum_count", "must not exceed maximum_count")
        if self.required and self.minimum_count == 0:
            raise validation_error(contract, "minimum_count", "must be positive for a required observation")
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "encodings", tuple(sorted(set(encodings), key=lambda value: value.value)))
        object.__setattr__(self, "color_spaces", tuple(sorted(set(colors), key=lambda value: value.value)))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ProprioceptiveObservationSpec:
    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtype: str
    units: tuple[str, ...]
    required: bool = True
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        _non_empty(self.name, contract, "name")
        shapes = _normalize_shapes(self.shapes, contract)
        if any(len(shape) != 1 for shape in shapes):
            raise validation_error(contract, "shapes", "proprioceptive shapes must be one-dimensional")
        dtype = _normalize_dtype(self.dtype, contract)
        units = tuple(self.units)
        if not units or any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise validation_error(contract, "units", "must contain non-empty strings")
        if any(shape[0] != len(units) for shape in shapes):
            raise validation_error(contract, "units", "length must match every permitted shape")
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    images: tuple[ImageObservationSpec, ...] = ()
    proprioception: tuple[ProprioceptiveObservationSpec, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        images = tuple(self.images)
        proprioception = tuple(self.proprioception)
        if any(not isinstance(value, ImageObservationSpec) for value in images):
            raise validation_error(contract, "images", "must contain ImageObservationSpec values")
        if any(not isinstance(value, ProprioceptiveObservationSpec) for value in proprioception):
            raise validation_error(
                contract, "proprioception", "must contain ProprioceptiveObservationSpec values"
            )
        names = [value.name for value in images + proprioception]
        if len(names) != len(set(names)):
            raise validation_error(contract, "observations", "observation names must be unique")
        object.__setattr__(self, "images", tuple(sorted(images, key=lambda value: value.name)))
        object.__setattr__(self, "proprioception", tuple(sorted(proprioception, key=lambda value: value.name)))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ObservationRequirements:
    images: tuple[ImageObservationSpec, ...] = ()
    proprioception: tuple[ProprioceptiveObservationSpec, ...] = ()
    minimum_image_count: int = 0
    maximum_image_count: int | None = None
    minimum_proprioception_count: int = 0
    maximum_proprioception_count: int | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        spec = ObservationSpec(self.images, self.proprioception)
        _count(self.minimum_image_count, contract, "minimum_image_count")
        _count(self.minimum_proprioception_count, contract, "minimum_proprioception_count")
        for field_name in ("maximum_image_count", "maximum_proprioception_count"):
            value = getattr(self, field_name)
            if value is not None:
                _count(value, contract, field_name)
        if self.maximum_image_count is not None and self.minimum_image_count > self.maximum_image_count:
            raise validation_error(contract, "minimum_image_count", "must not exceed maximum_image_count")
        if (
            self.maximum_proprioception_count is not None
            and self.minimum_proprioception_count > self.maximum_proprioception_count
        ):
            raise validation_error(
                contract, "minimum_proprioception_count", "must not exceed maximum_proprioception_count"
            )
        object.__setattr__(self, "images", spec.images)
        object.__setattr__(self, "proprioception", spec.proprioception)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class BenchmarkCapabilities:
    component_name: str
    component_version: str
    contract_version: str
    observation_spec: ObservationSpec
    action_spec: ActionSpec
    signal_registry: SignalRegistry
    supports_seeded_reset: bool
    supports_dynamic_instructions: bool
    supports_privileged_evaluation: bool
    task_suites: tuple[str, ...]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for field_name in ("component_name", "component_version", "contract_version"):
            _non_empty(getattr(self, field_name), contract, field_name)
        if not isinstance(self.observation_spec, ObservationSpec):
            raise validation_error(contract, "observation_spec", "must be an ObservationSpec")
        if not isinstance(self.action_spec, ActionSpec):
            raise validation_error(contract, "action_spec", "must be an ActionSpec")
        if not isinstance(self.signal_registry, SignalRegistry):
            raise validation_error(contract, "signal_registry", "must be a SignalRegistry")
        suites = tuple(self.task_suites)
        if not suites or any(not isinstance(value, str) or not value.strip() for value in suites):
            raise validation_error(contract, "task_suites", "must contain non-empty suite names")
        if len(suites) != len(set(suites)):
            raise validation_error(contract, "task_suites", "must not contain duplicates")
        object.__setattr__(self, "task_suites", tuple(sorted(suites)))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    component_name: str
    component_version: str
    contract_version: str
    observation_requirements: ObservationRequirements
    output_action_spec: ActionSpec
    supports_single_action: bool
    supports_action_chunks: bool
    minimum_action_horizon: int
    maximum_action_horizon: int
    supports_dynamic_instructions: bool
    supports_deterministic_reset: bool
    exposes_raw_policy_output: bool
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for field_name in ("component_name", "component_version", "contract_version"):
            _non_empty(getattr(self, field_name), contract, field_name)
        if not isinstance(self.observation_requirements, ObservationRequirements):
            raise validation_error(contract, "observation_requirements", "must be ObservationRequirements")
        if not isinstance(self.output_action_spec, ActionSpec):
            raise validation_error(contract, "output_action_spec", "must be an ActionSpec")
        if not self.supports_single_action and not self.supports_action_chunks:
            raise validation_error(
                contract, "supports_single_action/supports_action_chunks", "at least one mode must be supported"
            )
        for field_name in ("minimum_action_horizon", "maximum_action_horizon"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise validation_error(contract, field_name, "must be a positive integer")
        if self.minimum_action_horizon > self.maximum_action_horizon:
            raise validation_error(contract, "minimum_action_horizon", "must not exceed maximum_action_horizon")
        if self.supports_single_action and self.minimum_action_horizon > 1:
            raise validation_error(contract, "minimum_action_horizon", "must include horizon 1 for single actions")
        if not self.supports_action_chunks and (
            self.minimum_action_horizon != 1 or self.maximum_action_horizon != 1
        ):
            raise validation_error(contract, "maximum_action_horizon", "single-action-only policies require horizon 1")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
