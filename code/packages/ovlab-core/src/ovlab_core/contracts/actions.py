"""Explicit action-space and policy/action lifecycle contracts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .errors import validation_error
from .identifiers import PredictionId, StepId
from .metadata import Metadata, immutable_numeric_array, normalize_contract_value, normalize_metadata
from .time import validate_timestamp_ns


class ActionRepresentation(str, Enum):
    DELTA_POSE = "delta_pose"
    ABSOLUTE_POSE = "absolute_pose"
    JOINT_POSITION = "joint_position"
    JOINT_DELTA = "joint_delta"
    OTHER = "other"


class RotationRepresentation(str, Enum):
    AXIS_ANGLE = "axis_angle"
    EULER_XYZ = "euler_xyz"
    QUATERNION_XYZW = "quaternion_xyzw"
    QUATERNION_WXYZ = "quaternion_wxyz"
    NONE = "none"


class GripperConvention(str, Enum):
    OPEN_POSITIVE = "open_positive"
    CLOSED_POSITIVE = "closed_positive"
    BINARY_OPEN_ONE = "binary_open_one"
    BINARY_CLOSED_ONE = "binary_closed_one"
    NONE = "none"


class PredictionValidity(str, Enum):
    VALID = "valid"
    INVALID_SHAPE = "invalid_shape"
    INVALID_VALUE = "invalid_value"
    DECODE_ERROR = "decode_error"
    MODEL_ERROR = "model_error"
    UNAVAILABLE = "unavailable"


def _indices(value: tuple[int, ...], dimension: int, contract: str, field_name: str) -> tuple[int, ...]:
    result = tuple(value)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in result):
        raise validation_error(contract, field_name, "indices must be integers")
    if len(set(result)) != len(result):
        raise validation_error(contract, field_name, "indices must not contain duplicates")
    if any(index < 0 or index >= dimension for index in result):
        raise validation_error(contract, field_name, f"indices must be within [0, {dimension})")
    return result


@dataclass(frozen=True, slots=True)
class ActionSpec:
    dimension: int
    representation: ActionRepresentation
    translation_indices: tuple[int, ...] = ()
    rotation_indices: tuple[int, ...] = ()
    gripper_indices: tuple[int, ...] = ()
    rotation_representation: RotationRepresentation = RotationRepresentation.NONE
    gripper_convention: GripperConvention = GripperConvention.NONE
    units: tuple[str, ...] = ()
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    dtype: str = "float32"
    control_frequency_hz: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise validation_error(contract, "dimension", "must be a positive integer")
        if not isinstance(self.representation, ActionRepresentation):
            raise validation_error(contract, "representation", "must be an ActionRepresentation")
        translation = _indices(self.translation_indices, self.dimension, contract, "translation_indices")
        rotation = _indices(self.rotation_indices, self.dimension, contract, "rotation_indices")
        gripper = _indices(self.gripper_indices, self.dimension, contract, "gripper_indices")
        all_indices = translation + rotation + gripper
        if len(set(all_indices)) != len(all_indices):
            raise validation_error(contract, "indices", "semantic index groups must not overlap")
        if not isinstance(self.rotation_representation, RotationRepresentation):
            raise validation_error(contract, "rotation_representation", "must be a RotationRepresentation")
        if bool(rotation) == (self.rotation_representation is RotationRepresentation.NONE):
            raise validation_error(
                contract, "rotation_representation", "must be NONE exactly when rotation_indices is empty"
            )
        if not isinstance(self.gripper_convention, GripperConvention):
            raise validation_error(contract, "gripper_convention", "must be a GripperConvention")
        if bool(gripper) == (self.gripper_convention is GripperConvention.NONE):
            raise validation_error(
                contract, "gripper_convention", "must be NONE exactly when gripper_indices is empty"
            )
        units = tuple(self.units) if self.units else tuple("unitless" for _ in range(self.dimension))
        if len(units) != self.dimension:
            raise validation_error(contract, "units", "length must match dimension")
        if any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise validation_error(contract, "units", "entries must be non-empty strings")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise validation_error(contract, "dtype", "must not be empty or whitespace-only")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise validation_error(contract, "dtype", "must name a NumPy dtype") from exc
        if not np.issubdtype(dtype, np.number):
            raise validation_error(contract, "dtype", "must be numeric")
        minimum = self._limit(self.minimum, "minimum")
        maximum = self._limit(self.maximum, "maximum")
        if (minimum is None) != (maximum is None):
            raise validation_error(contract, "minimum/maximum", "must either both be provided or both be omitted")
        if minimum is not None and maximum is not None and np.any(minimum > maximum):
            raise validation_error(contract, "minimum/maximum", "minimum must be less than or equal to maximum")
        if self.control_frequency_hz is not None:
            if (
                isinstance(self.control_frequency_hz, bool)
                or not isinstance(self.control_frequency_hz, (int, float))
                or not np.isfinite(self.control_frequency_hz)
                or self.control_frequency_hz <= 0
            ):
                raise validation_error(contract, "control_frequency_hz", "must be a positive finite number")
        object.__setattr__(self, "translation_indices", translation)
        object.__setattr__(self, "rotation_indices", rotation)
        object.__setattr__(self, "gripper_indices", gripper)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))

    def _limit(self, value: np.ndarray | None, field_name: str) -> np.ndarray | None:
        if value is None:
            return None
        result = immutable_numeric_array(value, type(self).__name__, field_name, ndim=1)
        if result.shape != (self.dimension,):
            raise validation_error(type(self).__name__, field_name, f"shape must be ({self.dimension},)")
        return result


@dataclass(frozen=True, slots=True)
class RawPolicyOutput:
    prediction_id: PredictionId
    value: Any
    timestamp_ns: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.prediction_id, PredictionId):
            raise validation_error(contract, "prediction_id", "must be a PredictionId")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        object.__setattr__(self, "value", normalize_contract_value(self.value, contract, "value"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    prediction_id: PredictionId
    step_id: StepId
    actions: np.ndarray
    action_spec: ActionSpec
    timestamp_ns: int
    inference_duration_ns: int
    horizon: int
    validity: PredictionValidity = PredictionValidity.VALID
    confidence: float | None = None
    raw_output: RawPolicyOutput | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.prediction_id, PredictionId):
            raise validation_error(contract, "prediction_id", "must be a PredictionId")
        if not isinstance(self.step_id, StepId):
            raise validation_error(contract, "step_id", "must be a StepId")
        if not isinstance(self.action_spec, ActionSpec):
            raise validation_error(contract, "action_spec", "must be an ActionSpec")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        validate_timestamp_ns(self.inference_duration_ns, contract, "inference_duration_ns")
        if not isinstance(self.validity, PredictionValidity):
            raise validation_error(contract, "validity", "must be a PredictionValidity")
        source = np.asarray(self.actions)
        if source.ndim == 1:
            source = source[np.newaxis, :]
        actions = immutable_numeric_array(source, contract, "actions", ndim=2)
        if actions.shape[1] != self.action_spec.dimension:
            raise validation_error(
                contract, "actions", f"action dimension must equal ActionSpec.dimension ({self.action_spec.dimension})"
            )
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise validation_error(contract, "horizon", "must be a positive integer")
        if self.horizon != actions.shape[0]:
            raise validation_error(contract, "horizon", "must equal the first actions dimension")
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not np.isfinite(self.confidence)
                or not 0 <= self.confidence <= 1
            ):
                raise validation_error(contract, "confidence", "must be a finite value within [0, 1]")
        if self.raw_output is not None:
            if not isinstance(self.raw_output, RawPolicyOutput):
                raise validation_error(contract, "raw_output", "must be a RawPolicyOutput or None")
            if self.raw_output.prediction_id != self.prediction_id:
                raise validation_error(contract, "raw_output", "prediction_id must match")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))


@dataclass(frozen=True, slots=True)
class ExecutedAction:
    prediction_id: PredictionId
    step_id: StepId
    selected_chunk_index: int
    requested_action: np.ndarray
    applied_action: np.ndarray
    timestamp_ns: int
    modification_reason: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.prediction_id, PredictionId):
            raise validation_error(contract, "prediction_id", "must be a PredictionId")
        if not isinstance(self.step_id, StepId):
            raise validation_error(contract, "step_id", "must be a StepId")
        if (
            isinstance(self.selected_chunk_index, bool)
            or not isinstance(self.selected_chunk_index, int)
            or self.selected_chunk_index < 0
        ):
            raise validation_error(contract, "selected_chunk_index", "must be a non-negative integer")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        requested = immutable_numeric_array(self.requested_action, contract, "requested_action", ndim=1)
        applied = immutable_numeric_array(self.applied_action, contract, "applied_action", ndim=1)
        if requested.shape != applied.shape:
            raise validation_error(contract, "applied_action", "shape must match requested_action")
        modified = not np.array_equal(requested, applied)
        if modified and (not isinstance(self.modification_reason, str) or not self.modification_reason.strip()):
            raise validation_error(
                contract, "modification_reason", "must explain why requested and applied actions differ"
            )
        if self.modification_reason is not None and not isinstance(self.modification_reason, str):
            raise validation_error(contract, "modification_reason", "must be a string or None")
        object.__setattr__(self, "requested_action", requested)
        object.__setattr__(self, "applied_action", applied)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
