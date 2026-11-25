"""Verified OSC_POSE command convention used by pinned LIBERO evaluation."""

import numpy as np

from ovlab_core.contracts import (
    ActionRepresentation,
    ActionSpec,
    GripperConvention,
    RotationRepresentation,
)

from .errors import LiberoActionError, LiberoDependencyError


def libero_action_spec() -> ActionSpec:
    return ActionSpec(
        dimension=7,
        representation=ActionRepresentation.DELTA_POSE,
        translation_indices=(0, 1, 2),
        rotation_indices=(3, 4, 5),
        gripper_indices=(6,),
        rotation_representation=RotationRepresentation.AXIS_ANGLE,
        gripper_convention=GripperConvention.CLOSED_POSITIVE,
        units=("normalized_command",) * 7,
        minimum=np.full(7, -1.0, dtype=np.float32),
        maximum=np.full(7, 1.0, dtype=np.float32),
        dtype="float32",
        control_frequency_hz=20.0,
        metadata={"controller": "OSC_POSE", "gripper_open": -1.0, "gripper_closed": 1.0},
    )


def validate_runtime_action_spec(native_spec: object, expected: ActionSpec) -> None:
    try:
        low, high = native_spec
        low = np.asarray(low)
        high = np.asarray(high)
    except (TypeError, ValueError) as exc:
        raise LiberoDependencyError("native env.action_spec must be a (minimum, maximum) pair") from exc
    if low.shape != (expected.dimension,) or high.shape != (expected.dimension,):
        raise LiberoDependencyError(
            f"native action dimension differs from verified OSC_POSE dimension {expected.dimension}"
        )
    if not np.array_equal(low, expected.minimum) or not np.array_equal(high, expected.maximum):
        raise LiberoDependencyError("native action bounds differ from verified normalized [-1, 1] commands")


def validate_action(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    value = np.asarray(action)
    if value.shape != (spec.dimension,):
        raise LiberoActionError(f"LIBERO requires one action with shape ({spec.dimension},)")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise LiberoActionError("LIBERO action must contain finite numeric values")
    if value.dtype != np.dtype(spec.dtype):
        raise LiberoActionError(f"LIBERO action dtype must be {spec.dtype}, got {value.dtype}")
    value = value.copy()
    assert spec.minimum is not None and spec.maximum is not None
    if np.any(value < spec.minimum) or np.any(value > spec.maximum):
        raise LiberoActionError("LIBERO action is outside the declared normalized [-1, 1] range")
    return value


def settling_action() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
