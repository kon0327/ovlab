"""Model-independent mapping of native LIBERO observations."""

from collections.abc import Mapping

import numpy as np

from ovlab_core.contracts import (
    ColorSpace,
    ImageEncoding,
    ImageObservation,
    ImageObservationSpec,
    Instruction,
    ObservationSpec,
    PolicyObservation,
    ProprioceptiveObservation,
    ProprioceptiveObservationSpec,
    StepId,
)

from .errors import LiberoObservationError
from .settings import LiberoAdapterSettings, LiberoObservationProfile

_PROPRIO_SPECS = (
    ProprioceptiveObservationSpec("robot.eef.position", ((3,),), "float32", ("m",) * 3),
    ProprioceptiveObservationSpec("robot.eef.orientation_xyzw", ((4,),), "float32", ("unitless",) * 4),
    ProprioceptiveObservationSpec("robot.gripper.joint_position", ((2,),), "float32", ("rad",) * 2),
)
_OFT_PROPRIO_SPEC = ProprioceptiveObservationSpec(
    "robot.proprioception", ((8,),), "float32", ("m",) * 3 + ("rad",) * 5,
    metadata={
        "components": [
            "robot0_eef_pos[0:3]", "quat2axisangle(robot0_eef_quat)[0:3]",
            "robot0_gripper_qpos[0:2]",
        ],
        "convention": "LIBERO OpenVLA-OFT native state vector",
    },
)


def configured_cameras(settings: LiberoAdapterSettings) -> tuple[tuple[str, str], ...]:
    cameras = ((settings.camera_names[0], "camera.primary.rgb"),)
    if settings.observation_profile in (LiberoObservationProfile.DUAL_RGB, LiberoObservationProfile.NATIVE_OFT):
        cameras += ((settings.camera_names[1], "camera.wrist.rgb"),)
    return cameras


def observation_spec(settings: LiberoAdapterSettings) -> ObservationSpec:
    shape = (settings.camera_height, settings.camera_width, 3)
    images = tuple(
        ImageObservationSpec(canonical, (shape,), "uint8", (ImageEncoding.RAW,), (ColorSpace.RGB,))
        for _, canonical in configured_cameras(settings)
    )
    if settings.observation_profile is LiberoObservationProfile.RGB_PROPRIOCEPTION:
        proprioception = _PROPRIO_SPECS
    elif settings.observation_profile is LiberoObservationProfile.NATIVE_OFT:
        proprioception = (_OFT_PROPRIO_SPEC,)
    else:
        proprioception = ()
    return ObservationSpec(images, proprioception, {"image_transform": "rotate_180"})


def _required_array(raw: Mapping[str, object], key: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    if key not in raw:
        raise LiberoObservationError(f"required native observation {key!r} is missing")
    value = np.asarray(raw[key])
    if value.shape != shape:
        raise LiberoObservationError(f"native observation {key!r} has shape {value.shape}, expected {shape}")
    if value.dtype != np.dtype(dtype):
        raise LiberoObservationError(f"native observation {key!r} has dtype {value.dtype}, expected {dtype}")
    if not np.all(np.isfinite(value)):
        raise LiberoObservationError(f"native observation {key!r} contains non-finite values")
    return value


def _required_numeric_vector(raw: Mapping[str, object], key: str, size: int) -> np.ndarray:
    if key not in raw:
        raise LiberoObservationError(f"required native observation {key!r} is missing")
    value = np.asarray(raw[key])
    if value.shape != (size,) or value.dtype.kind not in "fc" or not np.all(np.isfinite(value)):
        raise LiberoObservationError(
            f"native observation {key!r} must be a finite numeric vector with shape ({size},)"
        )
    return value


def _quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Match the pinned OFT/Robosuite xyzw quaternion mapping without mutating its input."""
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(
        quaternion[:3] * (2.0 * np.arccos(quaternion[3]) / denominator), dtype=np.float32,
    )


def map_observation(
    raw: Mapping[str, object],
    settings: LiberoAdapterSettings,
    step_id: StepId,
    instruction: Instruction,
    timestamp_ns: int,
) -> PolicyObservation:
    shape = (settings.camera_height, settings.camera_width, 3)
    images = []
    for native_camera, canonical_name in configured_cameras(settings):
        native_key = f"{native_camera}_image"
        image = _required_array(raw, native_key, shape, "uint8")
        transformed = np.ascontiguousarray(image[::-1, ::-1])
        images.append(
            ImageObservation(
                canonical_name,
                transformed,
                timestamp_ns,
                ImageEncoding.RAW,
                ColorSpace.RGB,
                native_camera,
                {"native_key": native_key, "transform": "rotate_180"},
            )
        )
    proprioception = []
    if settings.observation_profile is LiberoObservationProfile.RGB_PROPRIOCEPTION:
        mappings = (
            ("robot0_eef_pos", _PROPRIO_SPECS[0]),
            ("robot0_eef_quat", _PROPRIO_SPECS[1]),
            ("robot0_gripper_qpos", _PROPRIO_SPECS[2]),
        )
        for native_key, spec in mappings:
            value = _required_array(raw, native_key, spec.shapes[0], spec.dtype)
            proprioception.append(
                ProprioceptiveObservation(spec.name, value, timestamp_ns, spec.units, {"native_key": native_key})
            )
    elif settings.observation_profile is LiberoObservationProfile.NATIVE_OFT:
        position = _required_numeric_vector(raw, "robot0_eef_pos", 3)
        quaternion = _required_numeric_vector(raw, "robot0_eef_quat", 4)
        gripper = _required_numeric_vector(raw, "robot0_gripper_qpos", 2)
        state = np.ascontiguousarray(np.concatenate(
            (position, _quat_xyzw_to_axis_angle(quaternion), gripper), dtype=np.float32,
        ))
        proprioception.append(ProprioceptiveObservation(
            _OFT_PROPRIO_SPEC.name,
            state,
            timestamp_ns,
            _OFT_PROPRIO_SPEC.units,
            {
                "native_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "mapping": "position_xyz+orientation_axis_angle_xyz+gripper_qpos_2",
            },
        ))
    return PolicyObservation(step_id, timestamp_ns, instruction, tuple(images), tuple(proprioception))
