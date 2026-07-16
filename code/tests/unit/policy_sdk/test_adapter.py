"""Policy adapter lifecycle and output validation tests."""

import inspect

import pytest

from helpers.contexts import make_episode_context, make_run_context
from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from ovlab_core.contracts import AdapterState
from ovlab_policy_sdk import PolicyAdapter, PolicyLifecycleError


def observation_for(episode):
    benchmark = MockBenchmark()
    benchmark.initialize(make_run_context())
    return benchmark.reset_episode(episode).initial_observation


def test_capabilities_are_unavailable_before_initialize() -> None:
    with pytest.raises(PolicyLifecycleError, match="not initialized"):
        _ = MockPolicy().capabilities


def test_predict_before_reset_fails_and_accepts_only_policy_observation() -> None:
    policy = MockPolicy()
    policy.initialize(make_run_context())
    with pytest.raises(PolicyLifecycleError):
        policy.predict(observation_for(make_episode_context()))
    assert tuple(inspect.signature(PolicyAdapter.predict).parameters) == ("self", "observation")


@pytest.mark.parametrize("horizon", [1, 3])
def test_prediction_matches_declared_horizon_and_is_deterministic(horizon: int) -> None:
    run = make_run_context()
    episode = make_episode_context(seed=13)
    observation = observation_for(episode)
    policy = MockPolicy(horizon=horizon)
    policy.initialize(run)
    policy.reset_episode(episode)
    first = policy.predict(observation)
    policy.end_episode(episode)
    policy.reset_episode(episode)
    second = policy.predict(observation)
    assert first.horizon == horizon
    assert first.actions.shape == (horizon, 3)
    assert first.action_spec is policy.capabilities.output_action_spec
    assert first.prediction_id == second.prediction_id
    assert (first.actions == second.actions).all()


def test_invalid_transitions_and_episode_ids_fail() -> None:
    run = make_run_context()
    episode = make_episode_context()
    policy = MockPolicy()
    with pytest.raises(PolicyLifecycleError):
        policy.reset_episode(episode)
    policy.initialize(run)
    with pytest.raises(PolicyLifecycleError, match="run_id"):
        policy.reset_episode(make_episode_context(run_id="other-run"))
    policy.reset_episode(episode)
    with pytest.raises(PolicyLifecycleError, match="does not match"):
        policy.end_episode(make_episode_context(episode_id="other-episode"))


def test_close_is_idempotent_and_blocks_operations() -> None:
    policy = MockPolicy()
    policy.initialize(make_run_context())
    policy.close()
    policy.close()
    assert policy.state is AdapterState.CLOSED
    with pytest.raises(PolicyLifecycleError):
        policy.reset_episode(make_episode_context())
