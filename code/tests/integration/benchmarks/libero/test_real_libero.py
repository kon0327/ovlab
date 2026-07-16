"""Minimal opt-in smoke test for the real pinned LIBERO runtime."""

import os

import numpy as np
import pytest

from helpers.contexts import make_run_context, make_step_context
from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_benchmarks.libero import LiberoAdapterSettings, LiberoBenchmarkAdapter
from ovlab_core.contracts import (
    EpisodeContext,
    EpisodeId,
    Instruction,
    InstructionId,
    InstructionSource,
    PredictionId,
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
        assert reset.initial_observation.images[0].data.shape == (64, 64, 3)
        assert capabilities.action_spec.dimension == 7
        request = BenchmarkActionRequest(
            make_step_context(episode, 0, reset.timestamp_ns + 1),
            PredictionId("real-libero-prediction"),
            0,
            np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            reset.timestamp_ns + 1,
        )
        result = adapter.step(request)
        assert {signal.name for signal in result.evaluation_signals} >= {
            "benchmark.task_success",
            "benchmark.reward",
        }
    finally:
        adapter.close()
