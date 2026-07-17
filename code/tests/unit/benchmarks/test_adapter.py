"""Benchmark adapter lifecycle and boundary tests."""

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context, make_step_context
from helpers.mock_benchmark import MockBenchmark
from ovlab_benchmarks import BenchmarkActionRequest, BenchmarkLifecycleError
from ovlab_core.contracts import AdapterState, PredictionId, SignalAccess


def action_request(episode, step_index: int = 0, action=None) -> BenchmarkActionRequest:
    if action is None:
        action = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    return BenchmarkActionRequest(
        make_step_context(episode, step_index, step_index * 10),
        PredictionId(f"prediction-{step_index}"),
        0,
        action,
        step_index * 10 + 1,
    )


def test_capabilities_are_unavailable_before_initialize() -> None:
    with pytest.raises(BenchmarkLifecycleError, match="not initialized"):
        _ = MockBenchmark().capabilities


def test_initialize_and_task_listing_are_deterministic() -> None:
    adapter = MockBenchmark()
    capabilities = adapter.initialize(make_run_context())
    assert adapter.state is AdapterState.READY
    assert capabilities.component_name == "mock-benchmark"
    assert adapter.list_tasks() == adapter.list_tasks()
    assert adapter.list_tasks()[0].task_id == make_episode_context().task_id


def test_reset_exposes_only_policy_observations_and_registered_signals() -> None:
    adapter = MockBenchmark()
    adapter.initialize(make_run_context())
    result = adapter.reset_episode(make_episode_context())
    assert adapter.state is AdapterState.EPISODE_ACTIVE
    assert {image.name for image in result.initial_observation.images} == {"front_rgb"}
    assert {value.name for value in result.evaluation_signals} == {"benchmark.task_success", "hidden_target"}
    registry = adapter.capabilities.signal_registry
    assert registry.resolve("hidden_target").access is SignalAccess.PRIVILEGED
    assert "hidden_target" not in {item.name for item in result.initial_observation.images}


def test_step_preserves_requested_and_applied_action_distinction() -> None:
    episode = make_episode_context()
    adapter = MockBenchmark(modify_actions=True)
    adapter.initialize(make_run_context())
    adapter.reset_episode(episode)
    result = adapter.step(action_request(episode))
    np.testing.assert_array_equal(
        result.executed_action.requested_action,
        np.array([0.1, 0.2, -0.1], dtype=np.float32),
    )
    assert not np.array_equal(result.executed_action.requested_action, result.executed_action.applied_action)
    assert result.executed_action.modification_reason == "mock action modification"


def test_step_rejects_more_than_one_action_vector() -> None:
    episode = make_episode_context()
    adapter = MockBenchmark()
    adapter.initialize(make_run_context())
    adapter.reset_episode(episode)
    with pytest.raises(ValueError, match="exactly one action"):
        adapter.step(action_request(episode, action=np.zeros(6, dtype=np.float32)))


def test_terminal_step_returns_adapter_to_ready() -> None:
    episode = make_episode_context()
    adapter = MockBenchmark(maximum_steps=1)
    adapter.initialize(make_run_context())
    adapter.reset_episode(episode)
    result = adapter.step(action_request(episode))
    assert result.terminated and result.next_observation is None
    assert adapter.state is AdapterState.READY
    with pytest.raises(BenchmarkLifecycleError):
        adapter.step(action_request(episode, 1))


def test_invalid_lifecycle_transitions_and_ids_fail() -> None:
    adapter = MockBenchmark()
    episode = make_episode_context()
    with pytest.raises(BenchmarkLifecycleError):
        adapter.step(action_request(episode))
    adapter.initialize(make_run_context())
    with pytest.raises(BenchmarkLifecycleError, match="run_id"):
        adapter.reset_episode(make_episode_context(run_id="other-run"))
    adapter.reset_episode(episode)
    wrong = make_episode_context(episode_id="other-episode")
    with pytest.raises(BenchmarkLifecycleError, match="IDs"):
        adapter.step(action_request(wrong))


def test_close_is_idempotent_and_blocks_operations() -> None:
    adapter = MockBenchmark()
    adapter.initialize(make_run_context())
    adapter.close()
    adapter.close()
    assert adapter.state is AdapterState.CLOSED
    with pytest.raises(BenchmarkLifecycleError):
        adapter.list_tasks()
