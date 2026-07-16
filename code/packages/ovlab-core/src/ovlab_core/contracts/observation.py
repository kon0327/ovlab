"""Policy-visible image, proprioceptive, and composite observations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .errors import validation_error
from .identifiers import StepId
from .instruction import Instruction
from .metadata import Metadata, immutable_numeric_array, normalize_metadata
from .time import validate_timestamp_ns


class ImageEncoding(str, Enum):
    RAW = "raw"
    PNG = "png"
    JPEG = "jpeg"
    OTHER = "other"


class ColorSpace(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAYSCALE = "grayscale"
    DEPTH = "depth"
    OTHER = "other"


def _non_empty_string(value: str, contract: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise validation_error(contract, field_name, "must not be empty or whitespace-only")


@dataclass(frozen=True, slots=True)
class ImageObservation:
    name: str
    data: np.ndarray
    timestamp_ns: int
    encoding: ImageEncoding
    color_space: ColorSpace
    camera_name: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        _non_empty_string(self.name, contract, "name")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        if not isinstance(self.encoding, ImageEncoding):
            raise validation_error(contract, "encoding", "must be an ImageEncoding")
        if not isinstance(self.color_space, ColorSpace):
            raise validation_error(contract, "color_space", "must be a ColorSpace")
        _non_empty_string(self.camera_name, contract, "camera_name")
        data = immutable_numeric_array(self.data, contract, "data", ndim=(2, 3))
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ProprioceptiveObservation:
    name: str
    values: np.ndarray
    timestamp_ns: int
    units: tuple[str, ...]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        _non_empty_string(self.name, contract, "name")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        values = immutable_numeric_array(self.values, contract, "values", ndim=1)
        units = tuple(self.units)
        if len(units) != values.shape[0]:
            raise validation_error(contract, "units", "length must match the values dimension")
        if any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise validation_error(contract, "units", "entries must be non-empty strings")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Inputs visible to a policy; evaluation signals are intentionally absent."""

    step_id: StepId
    timestamp_ns: int
    instruction: Instruction
    images: tuple[ImageObservation, ...] = ()
    proprioception: tuple[ProprioceptiveObservation, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.step_id, StepId):
            raise validation_error(contract, "step_id", "must be a StepId")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        if not isinstance(self.instruction, Instruction):
            raise validation_error(contract, "instruction", "must be an Instruction")
        images = tuple(self.images)
        proprioception = tuple(self.proprioception)
        if any(not isinstance(image, ImageObservation) for image in images):
            raise validation_error(contract, "images", "must contain only ImageObservation values")
        if any(not isinstance(item, ProprioceptiveObservation) for item in proprioception):
            raise validation_error(
                contract, "proprioception", "must contain only ProprioceptiveObservation values"
            )
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "proprioception", proprioception)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
