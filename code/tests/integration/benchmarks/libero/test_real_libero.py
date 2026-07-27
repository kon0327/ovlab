"""Minimal opt-in smoke test for the real pinned LIBERO runtime."""

import os

import numpy as np
import pytest

from helpers.contexts import make_run_context, make_step_context
from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_benchmarks.libero import (
    LiberoAdapterSettings,
    LiberoBenchmarkAdapter,
    LiberoRendererBackend,
)
from ovlab_core.contracts import (
    ActionRepresentation,
    ColorSpace,
    EpisodeContext,
    EpisodeId,
    GripperConvention,
    Instruction,
    InstructionId,
    InstructionSource,
    PredictionId,
    RotationRepresentation,
)

pytestmark = [
    pytest.mark.libero,
    pytest.mark.gpu,
    pytest.mark.manual,
    pytest.mark.skipif(
        os.environ.get("OVLAB_RUN_LIBERO") != "1",
        reason="set OVLAB_RUN_LIBERO=1 to run the real pinned LIBERO smoke test",
    ),
]


def test_real_libero_minimal_reset_and_step() -> None:
    adapter = LiberoBenchmarkAdapter(
        LiberoAdapterSettings(
            suite_names=("LIBERO-Spatial",),
            task_indices=(0,),
            camera_width=64,
            camera_height=64,
            maximum_episode_steps=1,
            initialization_settling_steps=1,
        )
    )
    run = make_run_context(run_id="real-libero-smoke", seed=0)
    try:
        capabilities = adapter.initialize(run)
        task = adapter.list_tasks()[0]
        instruction = Instruction(
            InstructionId("real-libero-instruction"),
            task.natural_language_instruction,
            1,
            InstructionSource.BENCHMARK,
        )
        episode = EpisodeContext(run.run_id, task.task_id, EpisodeId("real-libero-episode"), 0, 0, instruction)
        reset = adapter.reset_episode(episode)
        assert adapter.settings.renderer.resolved_backend is LiberoRendererBackend.EGL
        assert adapter.settings.renderer.device_id == 0
        native = adapter._native_observation
        assert native is not None
        native_image = np.asarray(native["agentview_image"])
        assert native_image.shape == (64, 64, 3)
        assert native_image.dtype == np.uint8
        assert native_image.flags.c_contiguous
        image = reset.initial_observation.images[0]
        assert image.name == "camera.primary.rgb"
        assert image.camera_name == "agentview"
        assert image.color_space is ColorSpace.RGB
        assert image.metadata == {"native_key": "agentview_image", "transform": "rotate_180"}
        np.testing.assert_array_equal(image.data, native_image[::-1, ::-1])
        assert reset.metadata["suite"] == "LIBERO-Spatial"
        assert reset.metadata["task_id"] == "libero/spatial/0"
        assert reset.metadata["initial_state_index"] == 0

        action_spec = capabilities.action_spec
        assert action_spec.dimension == 7
        assert action_spec.representation is ActionRepresentation.DELTA_POSE
        assert action_spec.translation_indices == (0, 1, 2)
        assert action_spec.rotation_indices == (3, 4, 5)
        assert action_spec.rotation_representation is RotationRepresentation.AXIS_ANGLE
        assert action_spec.gripper_indices == (6,)
        assert action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
        assert action_spec.units == ("normalized_command",) * 7
        assert action_spec.dtype == "float32"
        assert action_spec.control_frequency_hz == 20.0
        np.testing.assert_array_equal(action_spec.minimum, np.full(7, -1, dtype=np.float32))
        np.testing.assert_array_equal(action_spec.maximum, np.full(7, 1, dtype=np.float32))
        request = BenchmarkActionRequest(
            make_step_context(episode, 0, reset.timestamp_ns + 1),
            PredictionId("real-libero-prediction"),
            0,
            np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            reset.timestamp_ns + 1,
        )
        result = adapter.step(request)
        assert result.truncated and not result.terminated and result.success is False
        np.testing.assert_array_equal(result.executed_action.requested_action, request.requested_action)
        np.testing.assert_array_equal(result.executed_action.applied_action, request.requested_action)
        assert {signal.name for signal in result.evaluation_signals} >= {
            "benchmark.task_success",
            "benchmark.reward",
        }
    finally:
        adapter.close()
