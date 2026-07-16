"""Verified NumPy-only conversion from decoded OpenVLA to LIBERO commands."""

from dataclasses import dataclass

import numpy as np

from ovlab_core.contracts import (
    ActionRepresentation,
    ActionSpec,
    GripperConvention,
    RotationRepresentation,
    immutable_numeric_array,
)

from .errors import OpenVlaActionCodecError


def libero_target_action_spec() -> ActionSpec:
    """Return the exact OSC_POSE action contract declared by LiberoBenchmarkAdapter."""
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


@dataclass(frozen=True, slots=True)
class OpenVlaDecodedAction:
    """Typed pre-codec action; the marker prevents accidental second conversion."""

    value: np.ndarray

    def __post_init__(self) -> None:
        try:
            value = immutable_numeric_array(self.value, type(self).__name__, "value", ndim=1)
        except Exception as exc:
            raise OpenVlaActionCodecError("decoded OpenVLA action must be a finite numeric vector") from exc
        if value.shape != (7,):
            raise OpenVlaActionCodecError(f"decoded OpenVLA action must have shape (7,), got {value.shape}")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class LiberoActionCodecConfig:
    codec_id: str = "openvla-decoded-to-libero-osc-pose"
    version: str = "1.0.0"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.codec_id != "openvla-decoded-to-libero-osc-pose" or self.version != "1.0.0":
            raise OpenVlaActionCodecError("unsupported target action codec")
        if not isinstance(self.threshold, (int, float)) or not np.isfinite(self.threshold):
            raise OpenVlaActionCodecError("threshold must be finite")
        if float(self.threshold) != 0.5:
            raise OpenVlaActionCodecError("pinned OpenVLA conversion requires threshold 0.5")

    @property
    def identifier(self) -> str:
        return f"{self.codec_id}@{self.version}"


class LiberoActionCodec:
    """Apply upstream normalize_gripper_action(..., binarize=True), then inversion."""

    def __init__(self, config: LiberoActionCodecConfig = LiberoActionCodecConfig()) -> None:
        self.config = config

    def encode(self, decoded: OpenVlaDecodedAction) -> np.ndarray:
        if not isinstance(decoded, OpenVlaDecodedAction):
            raise OpenVlaActionCodecError("codec accepts only OpenVlaDecodedAction (prevents double conversion)")
        source = decoded.value
        if np.any(source[:6] < -1.0) or np.any(source[:6] > 1.0):
            raise OpenVlaActionCodecError("decoded pose is outside target normalized [-1, 1] bounds")
        gripper = float(source[6])
        if not 0.0 <= gripper <= 1.0:
            raise OpenVlaActionCodecError("decoded gripper must use the source [0, 1] convention")
        result = np.asarray(source, dtype=np.float32).copy()
        result[6] = -np.sign(2.0 * gripper - 1.0)
        return immutable_numeric_array(result, type(self).__name__, "action", ndim=1)


def action_specs_match(first: ActionSpec, second: ActionSpec) -> bool:
    fields = (
        "dimension", "representation", "translation_indices", "rotation_indices", "gripper_indices",
        "rotation_representation", "gripper_convention", "units", "dtype", "control_frequency_hz",
    )
    if any(getattr(first, name) != getattr(second, name) for name in fields):
        return False
    return all(
        (getattr(first, name) is None and getattr(second, name) is None)
        or (getattr(first, name) is not None and getattr(second, name) is not None
            and np.array_equal(getattr(first, name), getattr(second, name)))
        for name in ("minimum", "maximum")
    )
