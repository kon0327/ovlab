"""Generic evaluation-signal specifications, values, and registry."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import validation_error
from .identifiers import StepId
from .metadata import Metadata, normalize_contract_value, normalize_metadata
from .time import validate_timestamp_ns


class SignalAccess(str, Enum):
    POLICY_VISIBLE = "policy_visible"
    EVALUATION_ONLY = "evaluation_only"
    PRIVILEGED = "privileged"


@dataclass(frozen=True, slots=True)
class SignalSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    units: str
    access: SignalAccess
    description: str
    optional: bool = False

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for name, value in (("name", self.name), ("dtype", self.dtype), ("description", self.description)):
            if not isinstance(value, str) or not value.strip():
                raise validation_error(contract, name, "must not be empty or whitespace-only")
        if not isinstance(self.units, str):
            raise validation_error(contract, "units", "must be a string")
        if not isinstance(self.access, SignalAccess):
            raise validation_error(contract, "access", "must be a SignalAccess")
        shape = tuple(self.shape)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise validation_error(contract, "shape", "dimensions must be non-negative integers")
        if not isinstance(self.optional, bool):
            raise validation_error(contract, "optional", "must be a boolean")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class SignalValue:
    name: str
    value: Any
    timestamp_ns: int
    source: str
    step_id: StepId | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.name, str) or not self.name.strip():
            raise validation_error(contract, "name", "must not be empty or whitespace-only")
        if not isinstance(self.source, str) or not self.source.strip():
            raise validation_error(contract, "source", "must not be empty or whitespace-only")
        if self.step_id is not None and not isinstance(self.step_id, StepId):
            raise validation_error(contract, "step_id", "must be a StepId or None")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        object.__setattr__(self, "value", normalize_contract_value(self.value, contract, "value"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True, init=False)
class SignalRegistry:
    specs: tuple[SignalSpec, ...]

    def __init__(self, specs: Iterable[SignalSpec] = ()) -> None:
        values = tuple(specs)
        if any(not isinstance(spec, SignalSpec) for spec in values):
            raise validation_error(type(self).__name__, "specs", "must contain only SignalSpec values")
        names = [spec.name for spec in values]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise validation_error(type(self).__name__, "specs", f"duplicate signal names: {', '.join(duplicates)}")
        object.__setattr__(self, "specs", tuple(sorted(values, key=lambda spec: spec.name)))

    def __iter__(self):
        return iter(self.specs)

    def resolve(self, name: str) -> SignalSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def has_required(self, name: str) -> bool:
        try:
            return not self.resolve(name).optional
        except KeyError:
            return False
