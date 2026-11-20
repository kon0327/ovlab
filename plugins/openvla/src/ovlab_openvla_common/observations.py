"""Canonical, NumPy-only image selection and validation."""

import numpy as np

from ovlab_core.contracts import ColorSpace, ImageEncoding, PolicyObservation

from .errors import OpenVlaObservationError


def select_canonical_rgb(observation: PolicyObservation, camera_name: str) -> np.ndarray:
    matches = tuple(image for image in observation.images if image.name == camera_name)
    if not matches:
        raise OpenVlaObservationError(f"required canonical camera {camera_name!r} is missing")
    if len(matches) != 1:
        raise OpenVlaObservationError(f"canonical camera {camera_name!r} is ambiguous")
    image = matches[0]
    if image.encoding is not ImageEncoding.RAW or image.color_space is not ColorSpace.RGB:
        raise OpenVlaObservationError("OpenVLA requires a raw RGB image")
    value = np.asarray(image.data)
    if value.ndim != 3 or value.shape[2] != 3:
        raise OpenVlaObservationError(f"OpenVLA requires HWC RGB shape, got {value.shape}")
    if value.dtype != np.uint8:
        raise OpenVlaObservationError(f"OpenVLA requires uint8 RGB, got {value.dtype}")
    # A copy prevents processors from mutating the immutable OVLAB observation.
    return np.array(value, copy=True, order="C")
