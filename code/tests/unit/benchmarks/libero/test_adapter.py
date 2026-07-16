"""Concrete LIBERO adapter lifecycle tests over the fake native boundary."""

import numpy as np
import pytest

from helpers.contexts import make_run_context, make_step_context
from helpers.fake_libero import FakeLiberoBackend, fake_libero_episode
from ovlab_benchmarks import BenchmarkActionRequest, BenchmarkLifecycleError
from ovlab_benchmarks.libero import (
    LiberoAdapterSettings,
    LiberoBenchmarkAdapter,
    LiberoConfigurationError,
    LiberoDependencyError,
)
from ovlab_core.contracts import PredictionId


def request(context, index=0, action=None):
    return BenchmarkActionRequest(
        make_step_context(context, index, 10 + index),
        PredictionId(f"prediction-{index}"),
        0,
        np.zeros(7, dtype=np.float32) if action is None else action,
        10 + index,
    )


def adapter(*, backend=None, **settings):
    defaults = {"camera_width": 5, "camera_height": 4, "initialization_settling_steps": 0}
    defaults.update(settings)
    return LiberoBenchmarkAdapter(LiberoAdapterSettings(**defaults), backend=backend or FakeLiberoBackend())


def test_reset_metadata_state_and_instruction_mapping() -> None:
    backend = FakeLiberoBackend(state_count=3)
    benchmark = adapter(backend=backend)
    benchmark.initialize(make_run_context())
    reset = benchmark.reset_episode(fake_libero_episode(rollout_index=5))
    assert reset.metadata["initial_state_index"] == 2
    assert reset.metadata["settling_steps"] == 0
    assert reset.initial_observation.instruction.text == "perform native task 0"
    assert backend.environments[0].selected_state == "state-0-2"
    observation_names = {image.name for image in reset.initial_observation.images}
    assert not observation_names & {value.name for value in reset.evaluation_signals}


def test_reset_rejects_non_authoritative_instruction() -> None:
    benchmark = adapter()
    benchmark.initialize(make_run_context())
    with pytest.raises(LiberoConfigurationError, match="authoritative"):
        benchmark.reset_episode(fake_libero_episode(instruction="different"))


def test_settling_steps_use_verified_open_gripper_noop() -> None:
    backend = FakeLiberoBackend()
    benchmark = adapter(backend=backend, initialization_settling_steps=2)
    benchmark.initialize(make_run_context())
    benchmark.reset_episode(fake_libero_episode())
    assert len(backend.environments[0].actions) == 2
    np.testing.assert_array_equal(backend.environments[0].actions[0], [0, 0, 0, 0, 0, 0, -1])


def test_requested_and_applied_action_are_preserved() -> None:
    benchmark = adapter()
    context = fake_libero_episode()
    benchmark.initialize(make_run_context())
    benchmark.reset_episode(context)
    action = np.array([0.1, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    result = benchmark.step(request(context, action=action))
    np.testing.assert_array_equal(result.executed_action.requested_action, action)
    np.testing.assert_array_equal(result.executed_action.applied_action, action)


def test_success_terminates_and_releases_environment() -> None:
    backend = FakeLiberoBackend(success_after=1)
    benchmark = adapter(backend=backend)
    context = fake_libero_episode()
    benchmark.initialize(make_run_context())
    benchmark.reset_episode(context)
    result = benchmark.step(request(context))
    assert result.terminated and not result.truncated and result.success is True
    assert backend.environments[0].closed
    with pytest.raises(BenchmarkLifecycleError):
        benchmark.step(request(context, 1))


def test_time_limit_truncates_without_fabricating_success() -> None:
    backend = FakeLiberoBackend()
    benchmark = adapter(backend=backend, maximum_episode_steps=1)
    context = fake_libero_episode()
    benchmark.initialize(make_run_context())
    benchmark.reset_episode(context)
    result = benchmark.step(request(context))
    assert result.truncated and not result.terminated and result.success is False
    assert backend.environments[0].closed


def test_close_releases_environment_and_is_idempotent() -> None:
    backend = FakeLiberoBackend()
    benchmark = adapter(backend=backend)
    benchmark.initialize(make_run_context())
    benchmark.reset_episode(fake_libero_episode())
    benchmark.close()
    benchmark.close()
    assert backend.environments[0].closed


def test_step_before_reset_fails() -> None:
    benchmark = adapter()
    context = fake_libero_episode()
    benchmark.initialize(make_run_context())
    with pytest.raises(BenchmarkLifecycleError):
        benchmark.step(request(context))


def test_default_backend_reports_missing_dependency_clearly() -> None:
    with pytest.raises(LiberoDependencyError):
        LiberoBenchmarkAdapter().initialize(make_run_context())
