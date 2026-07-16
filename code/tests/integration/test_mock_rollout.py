"""CPU-only end-to-end rollout across both adapter boundaries."""

import numpy as np

from helpers.contexts import make_episode_context, make_run_context, make_step_context
from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_core import negotiate_capabilities
from ovlab_core.contracts import AdapterState


def test_negotiated_mock_rollout_preserves_boundary_evidence() -> None:
    run = make_run_context(seed=19)
    episode = make_episode_context(seed=23)
    benchmark = MockBenchmark(maximum_steps=3, modify_actions=True)
    policy = MockPolicy(horizon=2)
    benchmark_caps = benchmark.initialize(run)
    policy_caps = policy.initialize(run)
    negotiate_capabilities(benchmark_caps, policy_caps).require_compatible()

    reset = benchmark.reset_episode(episode)
    policy.reset_episode(episode)
    observation = reset.initial_observation
    predictions = []
    executed_actions = []
    signals = list(reset.evaluation_signals)

    for step_index in range(3):
        prediction = policy.predict(observation)
        chunk_index = step_index % prediction.horizon
        request = BenchmarkActionRequest(
            make_step_context(episode, step_index, observation.timestamp_ns),
            prediction.prediction_id,
            chunk_index,
            prediction.actions[chunk_index],
            prediction.timestamp_ns,
        )
        result = benchmark.step(request)
        predictions.append(prediction)
        executed_actions.append(result.executed_action)
        signals.extend(result.evaluation_signals)
        if result.terminated or result.truncated:
            policy.end_episode(episode)
            break
        assert result.next_observation is not None
        observation = result.next_observation

    assert benchmark.state is AdapterState.READY
    assert policy.state is AdapterState.READY
    assert len(predictions) == len(executed_actions) == 3
    assert all(
        action.prediction_id == prediction.prediction_id
        for action, prediction in zip(executed_actions, predictions)
    )
    assert all(not np.array_equal(action.requested_action, action.applied_action) for action in executed_actions)
    assert {signal.name for signal in signals} == {"success", "hidden_target"}
    policy_inputs = reset.initial_observation.images + reset.initial_observation.proprioception
    policy_input_names = {item.name for item in policy_inputs}
    assert "hidden_target" not in policy_input_names
