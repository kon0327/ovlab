"""Configured benchmark guards preserve lifecycle recovery and locked task limits."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from helpers.contexts import make_run_context
from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_benchctl.benchmark_plugins import create_adapter
from ovlab_benchctl.errors import ConfigCompatibilityError
from ovlab_benchmarks import BenchmarkPlugin, TaskCatalog
from ovlab_core.contracts import AdapterState, BenchmarkIdentity, EpisodeTerminalStatus, TaskId
from ovlab_policy_sdk import PolicyInferenceError
from ovlab_runner import (
    DeterministicClock, EpisodeErrorPolicy, ExperimentRunner, InMemoryRunArtifactStore, RunnerState,
)


class _FixturePlugin(BenchmarkPlugin):
    def __init__(self, delegate):
        self.delegate = delegate
        probe = TrackingBenchmark(maximum_steps=delegate.maximum_steps, task_count=delegate.task_count,
                                  terminal_outcomes=delegate.terminal_outcomes)
        try:
            self.capabilities = probe.initialize(make_run_context())
            self.catalog = TaskCatalog("runtime-guard-fixture", "1", probe.list_tasks())
        finally:
            probe.close()

    def validate_benchmark(self, document, path):
        pass

    def resolve_settings(self, document, context):
        return None

    def identity(self, settings):
        return BenchmarkIdentity("fixture", "fixture", "1", "fixture-revision")

    def describe_capabilities(self, settings):
        return self.capabilities

    def task_catalog(self, settings):
        return self.catalog

    def create_adapter(self, settings):
        return self.delegate


class _ResourceBenchmark(TrackingBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resource_active = False
        self.abort_count = 0

    def _reset_episode(self, context):
        assert not self.resource_active, "the preceding episode still owns native resources"
        self.resource_active = True
        return super()._reset_episode(context)

    def _step(self, request):
        result = super()._step(request)
        if result.terminated or result.truncated:
            self.resource_active = False
        return result

    def _abort_episode(self):
        self.abort_count += 1
        self.resource_active = False


class _FailFirstEpisodePolicy(TrackingPolicy):
    def __init__(self):
        super().__init__()
        self.failed_once = False

    def _predict(self, observation):
        if not self.failed_once:
            self.failed_once = True
            raise PolicyInferenceError("the first episode fails after benchmark reset")
        return super()._predict(observation)


@pytest.mark.parametrize("failure_policy,task_count,rollouts", [
    (EpisodeErrorPolicy.CONTINUE_TASK, 1, 2),
    (EpisodeErrorPolicy.CONTINUE_RUN, 2, 1),
])
def test_registered_benchmark_recovers_after_failed_episode(failure_policy, task_count, rollouts):
    delegate = _ResourceBenchmark(task_count=task_count, terminal_outcomes=("success",) * task_count)
    adapter = create_adapter(SimpleNamespace(implementation=_FixturePlugin(delegate)), None)
    plan = runner_plan(
        selected_task_ids=tuple(TaskId(f"mock-task-{index}") for index in range(task_count)),
        rollout_count_per_task=rollouts, episode_error_policy=failure_policy,
    )
    store = InMemoryRunArtifactStore()
    runner = ExperimentRunner(plan, adapter, _FailFirstEpisodePolicy(), store, clock=DeterministicClock())
    try:
        runner.connect()
        runner.run()
        assert runner.state is RunnerState.COMPLETED
    finally:
        runner.close()

    traces = [episode["trace"] for episode in store.runs["runner-test"]["episodes"].values()]
    assert [trace.terminal_status for trace in traces] == [
        EpisodeTerminalStatus.POLICY_ERROR, EpisodeTerminalStatus.SUCCESS,
    ]
    assert runner.state is RunnerState.CLOSED
    assert delegate.abort_count == 1 and len(delegate.reset_contexts) == 2
    assert not delegate.resource_active
    assert adapter.state is AdapterState.CLOSED and delegate.state is AdapterState.CLOSED


@pytest.mark.parametrize("change", [{"maximum_steps": 9}, {"task_index": 7}])
def test_runtime_task_limit_and_index_must_match_locked_catalog(change):
    class DriftingBenchmark(TrackingBenchmark):
        def _list_tasks(self):
            return tuple(replace(task, **change) for task in super()._list_tasks())

    delegate = DriftingBenchmark()
    adapter = create_adapter(SimpleNamespace(implementation=_FixturePlugin(delegate)), None)
    try:
        adapter.initialize(make_run_context())
        with pytest.raises(ConfigCompatibilityError, match="runtime task identity differs from catalog"):
            adapter.list_tasks()
        assert delegate.reset_contexts == []
    finally:
        adapter.close()
