"""Action-chunk execution contracts."""

import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_runner import (
    ActionExecutionMode, ActionExecutionPolicy, DeterministicClock, ExperimentRunner,
    InMemoryRunArtifactStore,
)


@pytest.mark.parametrize(
    "policy,horizon,expected_indices,expected_predictions",
    [
        (ActionExecutionPolicy(ActionExecutionMode.RECEDING_HORIZON), 3, (0, 0, 0), 3),
        (ActionExecutionPolicy(ActionExecutionMode.OPEN_LOOP_CHUNK), 3, (0, 1, 2), 1),
        (ActionExecutionPolicy(ActionExecutionMode.FIXED_REPLAN_INTERVAL, 2), 3, (0, 1, 0), 2),
    ],
)
def test_chunk_modes_preserve_prediction_and_chunk_indices(policy, horizon, expected_indices, expected_predictions) -> None:
    store = InMemoryRunArtifactStore()
    plan = runner_plan(action_execution_policy=policy)
    runner = ExperimentRunner(plan, TrackingBenchmark(maximum_steps=3), TrackingPolicy(horizon=horizon), store, clock=DeterministicClock())
    runner.connect()
    runner.run()
    episode = next(iter(store.runs["runner-test"]["episodes"].values()))
    trace = episode["trace"]
    assert tuple(action.selected_chunk_index for action in trace.executed_actions) == expected_indices
    assert len(trace.policy_predictions) == expected_predictions
    assert len(trace.executed_actions) == 3
    predictions = {prediction.prediction_id: prediction for prediction in trace.policy_predictions}
    assert all(action.prediction_id in predictions for action in trace.executed_actions)
    assert all(
        action.requested_action.tobytes()
        == predictions[action.prediction_id].actions[action.selected_chunk_index].tobytes()
        for action in trace.executed_actions
    )
    assert all(action.metadata["closed_loop_step_duration_ns"] > 0 for action in trace.executed_actions)


def test_early_termination_discards_remaining_chunk() -> None:
    store = InMemoryRunArtifactStore()
    runner = ExperimentRunner(
        runner_plan(action_execution_policy=ActionExecutionPolicy(ActionExecutionMode.OPEN_LOOP_CHUNK)),
        TrackingBenchmark(maximum_steps=1), TrackingPolicy(horizon=3), store, clock=DeterministicClock(),
    )
    runner.connect(); runner.run()
    trace = next(iter(store.runs["runner-test"]["episodes"].values()))["trace"]
    assert len(trace.executed_actions) == 1
    assert trace.executed_actions[0].selected_chunk_index == 0
    assert trace.executed_actions[0].prediction_id == trace.policy_predictions[0].prediction_id
    assert not any(
        action.selected_chunk_index in (1, 2) for action in trace.executed_actions
    )
