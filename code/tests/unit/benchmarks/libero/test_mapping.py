"""Observation, action, and signal mapping tests."""

import numpy as np
import pytest

from helpers.contexts import make_episode_context
from helpers.fake_libero import fake_raw_observation
from ovlab_benchmarks.libero import LiberoActionError, LiberoAdapterSettings, LiberoObservationError, LiberoObservationProfile
from ovlab_benchmarks.libero.actions import libero_action_spec, validate_action
from ovlab_benchmarks.libero.observations import map_observation
from ovlab_benchmarks.libero.signals import map_signals, signal_registry
from ovlab_core.contracts import GripperConvention, SignalAccess, StepId


def settings(profile=LiberoObservationProfile.PRIMARY_RGB):
    return LiberoAdapterSettings(
        camera_width=5,
        camera_height=4,
        observation_profile=profile,
        initialization_settling_steps=0,
    )


def test_primary_image_is_rotated_180_degrees_without_resize() -> None:
    episode = make_episode_context()
    raw = fake_raw_observation()
    observation = map_observation(raw, settings(), StepId("step"), episode.initial_instruction, 2)
    assert observation.images[0].name == "camera.primary.rgb"
    np.testing.assert_array_equal(observation.images[0].data, raw["agentview_image"][::-1, ::-1])
    assert observation.images[0].data.dtype == np.uint8


def test_dual_camera_and_proprioception_profiles() -> None:
    episode = make_episode_context()
    raw = fake_raw_observation()
    dual = map_observation(
        raw, settings(LiberoObservationProfile.DUAL_RGB), StepId("dual"), episode.initial_instruction, 2
    )
    assert [image.name for image in dual.images] == ["camera.primary.rgb", "camera.wrist.rgb"]
    proprio = map_observation(
        raw,
        settings(LiberoObservationProfile.RGB_PROPRIOCEPTION),
        StepId("proprio"),
        episode.initial_instruction,
        2,
    )
    assert [item.name for item in proprio.proprioception] == [
        "robot.eef.position",
        "robot.eef.orientation_xyzw",
        "robot.gripper.joint_position",
    ]


def test_missing_key_wrong_shape_and_dtype_fail_clearly() -> None:
    episode = make_episode_context()
    raw = fake_raw_observation()
    del raw["agentview_image"]
    with pytest.raises(LiberoObservationError, match="missing"):
        map_observation(raw, settings(), StepId("missing"), episode.initial_instruction, 2)
    raw = fake_raw_observation()
    raw["agentview_image"] = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(LiberoObservationError, match="shape"):
        map_observation(raw, settings(), StepId("shape"), episode.initial_instruction, 2)
    raw = fake_raw_observation()
    raw["agentview_image"] = raw["agentview_image"].astype(np.float32)
    with pytest.raises(LiberoObservationError, match="dtype"):
        map_observation(raw, settings(), StepId("dtype"), episode.initial_instruction, 2)


def test_action_convention_and_validation() -> None:
    spec = libero_action_spec()
    assert spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
    valid = np.zeros(7, dtype=np.float32)
    np.testing.assert_array_equal(validate_action(valid, spec), valid)
    for invalid in (
        np.zeros(6, dtype=np.float32),
        np.array([0, 0, 0, 0, 0, 0, 2], dtype=np.float32),
        np.full(7, np.nan, dtype=np.float32),
        np.zeros(7, dtype=np.float64),
    ):
        with pytest.raises(LiberoActionError):
            validate_action(invalid, spec)


def test_signals_are_declared_stable_and_separate() -> None:
    profile = LiberoObservationProfile.PRIMARY_RGB
    registry = signal_registry(profile)
    values = map_signals(
        fake_raw_observation(),
        profile,
        StepId("step"),
        2,
        reward=0.5,
        success=False,
        terminated=False,
        truncated=False,
        native_step_index=1,
        initial_state_index=2,
    )
    assert [value.name for value in values] == sorted(value.name for value in values)
    assert {value.name for value in values} == {spec.name for spec in registry}
    assert registry.resolve("robot.eef.position").access is SignalAccess.PRIVILEGED
