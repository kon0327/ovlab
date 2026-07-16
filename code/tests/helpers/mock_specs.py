"""Shared deterministic specifications for adapter mocks."""

import numpy as np

from ovlab_core.contracts import (
    ActionRepresentation,
    ActionSpec,
    ColorSpace,
    GripperConvention,
    ImageEncoding,
    ImageObservationSpec,
    ObservationRequirements,
    ObservationSpec,
    ProprioceptiveObservationSpec,
    RotationRepresentation,
)


def mock_action_spec() -> ActionSpec:
    return ActionSpec(
        dimension=3,
        representation=ActionRepresentation.DELTA_POSE,
        translation_indices=(0, 1),
        gripper_indices=(2,),
        rotation_representation=RotationRepresentation.NONE,
        gripper_convention=GripperConvention.OPEN_POSITIVE,
        units=("m", "m", "unitless"),
        minimum=np.full(3, -1.0),
        maximum=np.full(3, 1.0),
        dtype="float32",
        control_frequency_hz=10.0,
    )


def mock_observation_spec() -> ObservationSpec:
    return ObservationSpec(
        images=(
            ImageObservationSpec(
                "front_rgb", ((4, 4, 3),), "uint8", (ImageEncoding.RAW,), (ColorSpace.RGB,)
            ),
        ),
        proprioception=(
            ProprioceptiveObservationSpec("robot_state", ((2,),), "float32", ("rad", "rad")),
        ),
    )


def mock_observation_requirements() -> ObservationRequirements:
    spec = mock_observation_spec()
    return ObservationRequirements(
        images=spec.images,
        proprioception=spec.proprioception,
        minimum_image_count=1,
        maximum_image_count=1,
        minimum_proprioception_count=1,
        maximum_proprioception_count=1,
    )
