"""Episode failure policies preserve finalized raw evidence."""

import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_core.contracts import EpisodeTerminalStatus
from ovlab_policy_sdk import PolicyInferenceError
from ovlab_runner import (
    DeterministicClock, EpisodeErrorPolicy, ExperimentExecutionError, ExperimentRunner,
    InMemoryRunArtifactStore, RunnerState,
)


class FailingPolicy(TrackingPolicy):
    def _predict(self, observation):
        raise PolicyInferenceError("synthetic inference failure")


def test_policy_error_persists_trace_without_fake_action_and_stop_run_fails() -> None:
    store = InMemoryRunArtifactStore()
    benchmark, policy = TrackingBenchmark(), FailingPolicy()
    runner = ExperimentRunner(runner_plan(), benchmark, policy, store, clock=DeterministicClock())
    runner.connect()
    with pytest.raises(ExperimentExecutionError): runner.run()
    trace = next(iter(store.runs["runner-test"]["episodes"].values()))["trace"]
    assert trace.terminal_status is EpisodeTerminalStatus.POLICY_ERROR
    assert trace.executed_actions == ()
    assert runner.state is RunnerState.FAILED and benchmark.closed and policy.closed
    assert store.write_order[-1] == "manifest.failed"


def test_continue_task_can_start_next_episode_after_policy_failure() -> None:
    store = InMemoryRunArtifactStore()
    plan = runner_plan(rollout_count_per_task=2, episode_error_policy=EpisodeErrorPolicy.CONTINUE_TASK)
    runner = ExperimentRunner(plan, TrackingBenchmark(), FailingPolicy(), store, clock=DeterministicClock())
    runner.connect(); runner.run()
    assert len(store.runs["runner-test"]["episodes"]) == 2


def test_continue_run_skips_remaining_rollouts_of_failed_task() -> None:
    store = InMemoryRunArtifactStore()
    plan = runner_plan(rollout_count_per_task=2, episode_error_policy=EpisodeErrorPolicy.CONTINUE_RUN)
    runner = ExperimentRunner(plan, TrackingBenchmark(), FailingPolicy(), store, clock=DeterministicClock())
    runner.connect(); runner.run()
    assert len(store.runs["runner-test"]["episodes"]) == 1
