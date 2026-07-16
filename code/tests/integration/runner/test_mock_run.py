"""End-to-end in-process mock execution and offline metric persistence."""

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_metrics import MetricEvaluator, MetricRegistry
from ovlab_runner import (
    DeterministicClock, ExperimentRunner, FilesystemRunArtifactStore,
    InMemoryRunArtifactStore, RunnerState,
)


def test_mock_run_persists_trace_before_metrics_and_recomputes_identically() -> None:
    store = InMemoryRunArtifactStore()
    benchmark, policy = TrackingBenchmark(maximum_steps=3), TrackingPolicy()
    runner = ExperimentRunner(runner_plan(rollout_count_per_task=2), benchmark, policy, store, clock=DeterministicClock())
    report = runner.connect()
    task_results = runner.run()
    assert runner.state is RunnerState.COMPLETED
    assert benchmark.closed and policy.closed
    run = store.runs["runner-test"]
    assert len(run["episodes"]) == 2
    assert store.write_order[-1] == "manifest.completed"
    for episode in run["episodes"].values():
        trace = episode["trace"]
        assert store.write_order.index(f"trace:{trace.episode_context.episode_id}") < store.write_order.index(f"episode-metrics:{trace.episode_context.episode_id}")
        selected = MetricRegistry(tuple(MetricRegistry.default().resolve(metric) for metric in runner.plan.enabled_metric_ids))
        assert MetricEvaluator(selected).evaluate(trace) == episode["metrics"]
    assert task_results["mock-task-0"]
    assert benchmark.reset_contexts == policy.reset_contexts


def test_filesystem_run_round_trip_includes_trace_metrics_and_final_manifest(tmp_path) -> None:
    store = FilesystemRunArtifactStore(tmp_path)
    benchmark, policy = TrackingBenchmark(maximum_steps=3), TrackingPolicy()
    runner = ExperimentRunner(runner_plan(), benchmark, policy, store, clock=DeterministicClock())
    runner.connect(); runner.run()
    context = benchmark.reset_contexts[0]
    trace = store.read_episode_trace(context.run_id, context.task_id, context.episode_id)
    metrics = store.read_metric_results(context.run_id, context.task_id, context.episode_id)
    assert len(trace.executed_actions) == 3
    assert metrics
    completed = tuple(tmp_path.glob("*/manifest.completed.json"))
    assert len(completed) == 1
