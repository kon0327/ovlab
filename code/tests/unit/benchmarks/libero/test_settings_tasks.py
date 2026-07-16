"""LIBERO settings, discovery, and deterministic selection tests."""

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from helpers.fake_libero import FakeLiberoBackend
from ovlab_benchmarks.libero import (
    InitialStateSelection,
    LiberoAdapterSettings,
    LiberoBenchmarkAdapter,
    LiberoConfigurationError,
    LiberoObservationProfile,
)
from ovlab_benchmarks.libero.tasks import select_initial_state_index


def test_discovery_preserves_native_order_and_stable_ids() -> None:
    settings = LiberoAdapterSettings(initialization_settling_steps=0)
    first = LiberoBenchmarkAdapter(settings, backend=FakeLiberoBackend())
    second = LiberoBenchmarkAdapter(settings, backend=FakeLiberoBackend())
    first.initialize(make_run_context())
    second.initialize(make_run_context())
    assert [str(task.task_id) for task in first.list_tasks()] == ["libero/spatial/0", "libero/spatial/1"]
    assert first.list_tasks() == second.list_tasks()


def test_unknown_suite_and_out_of_range_task_are_rejected() -> None:
    with pytest.raises(LiberoConfigurationError, match="unknown"):
        LiberoBenchmarkAdapter(
            LiberoAdapterSettings(suite_names=("unknown",)), backend=FakeLiberoBackend()
        ).initialize(make_run_context())
    with pytest.raises(LiberoConfigurationError, match="outside"):
        LiberoBenchmarkAdapter(
            LiberoAdapterSettings(task_indices=(9,)), backend=FakeLiberoBackend()
        ).initialize(make_run_context())


def test_settings_validate_dimensions_indices_and_profile_cameras() -> None:
    with pytest.raises(LiberoConfigurationError):
        LiberoAdapterSettings(camera_width=0)
    with pytest.raises(LiberoConfigurationError):
        LiberoAdapterSettings(task_indices=(1, 1))
    with pytest.raises(LiberoConfigurationError):
        LiberoAdapterSettings(
            camera_names=("agentview",), observation_profile=LiberoObservationProfile.DUAL_RGB
        )


def test_initial_state_selection_is_deterministic_without_global_rng_mutation() -> None:
    episode = make_episode_context(seed=47)
    state = np.random.get_state()
    assert select_initial_state_index(episode, 3, InitialStateSelection.ROLLOUT_INDEX, 0) == 0
    first = select_initial_state_index(episode, 3, InitialStateSelection.SEEDED, 5)
    second = select_initial_state_index(episode, 3, InitialStateSelection.SEEDED, 5)
    assert first == second
    current = np.random.get_state()
    assert state[0] == current[0] and np.array_equal(state[1], current[1])
