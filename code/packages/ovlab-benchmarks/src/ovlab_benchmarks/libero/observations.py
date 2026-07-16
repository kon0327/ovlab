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


def configured_cameras(settings: LiberoAdapterSettings) -> tuple[tuple[str, str], ...]:
    cameras = ((settings.camera_names[0], "camera.primary.rgb"),)
    if settings.observation_profile is LiberoObservationProfile.DUAL_RGB:
        cameras += ((settings.camera_names[1], "camera.wrist.rgb"),)
    return cameras


def observation_spec(settings: LiberoAdapterSettings) -> ObservationSpec:
    shape = (settings.camera_height, settings.camera_width, 3)
    images = tuple(
        ImageObservationSpec(canonical, (shape,), "uint8", (ImageEncoding.RAW,), (ColorSpace.RGB,))
        for _, canonical in configured_cameras(settings)
    )
    proprioception = (
        _PROPRIO_SPECS
        if settings.observation_profile is LiberoObservationProfile.RGB_PROPRIOCEPTION
        else ()
    )
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
    return PolicyObservation(step_id, timestamp_ns, instruction, tuple(images), tuple(proprioception))
