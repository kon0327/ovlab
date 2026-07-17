"""Gate A: dependency-light end-to-end execution and offline reproducibility."""

import json
from pathlib import Path

import numpy as np
import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from helpers.contexts import make_run_context
from ovlab_benchctl import ConfigResolver, load
from ovlab_core.contracts import EpisodeTerminalStatus, PolicyObservation, TaskContext, TaskId
from ovlab_metrics import (
    EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, MetricStatus, TaskMetricPlugin,
    aggregate_episode_results,
)
from ovlab_runner import (
    ArtifactError, DeterministicClock, ExperimentRunner, FilesystemRunArtifactStore,
    InMemoryRunArtifactStore, ProvenanceSnapshot, RunnerState, StaticProvenanceProvider,
)
from ovlab_runner.execution import make_episode_context


REPOSITORY = Path(__file__).resolve().parents[4]
CONFIGS = REPOSITORY / "configs"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OPENVLA_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"


def _profile(tmp_path: Path) -> Path:
    target = tmp_path / "gate-a-profile.yaml"
    target.write_text(
        f'''schema_version: "0.1.0"
kind: local_profile
id: gate-a-test

paths:
  checkpoint_root: {tmp_path}/checkpoints
  dataset_root: {tmp_path}/datasets
  runs_root: {tmp_path}/runs

devices:
  primary_gpu: cuda:0
''',
        encoding="utf-8",
    )
    return target


def _offline_episode_results(trace, plan, registry):
    plugins = tuple(
        registry.resolve(metric_id)
        for metric_id in plan.enabled_metric_ids
        if isinstance(registry.resolve(metric_id), EpisodeMetricPlugin)
    )
    configurations = {
        key: value for key, value in plan.metric_configurations.items()
        if key in {plugin.descriptor.metric_id for plugin in plugins}
    }
    return MetricEvaluator(MetricRegistry(plugins)).evaluate(trace, configurations)


def _offline_task_results(task, episode_results, plan, registry):
    context = TaskContext(
        plan.run_context.run_id, task.task_id, task.suite_name, task.task_name,
        task.task_index, task.metadata,
    )
    aggregated = []
    for metric_id in plan.enabled_metric_ids:
        plugin = registry.resolve(metric_id)
        if isinstance(plugin, TaskMetricPlugin):
            source_id = "task.success" if metric_id == "task.success_rate" else metric_id
            source = tuple(result for result in episode_results if result.metric_id == source_id)
            if source:
                aggregated.append(plugin.aggregate(
                    context, source, plan.metric_configurations.get(metric_id, plugin.default_config)
                ))
        else:
            source = tuple(result for result in episode_results if result.metric_id == metric_id)
            if source:
                aggregated.append(aggregate_episode_results(context, source))
    return tuple(aggregated)


def _scientific_signature(trace, metrics):
    return {
        "task_id": str(trace.episode_context.task_id),
        "rollout_index": trace.episode_context.rollout_index,
        "seed": trace.episode_context.seed,
        "instruction": trace.episode_context.initial_instruction.text,
        "terminal_status": trace.terminal_status.value,
        "predictions": tuple(
            (prediction.horizon, prediction.validity.value, prediction.actions.tobytes())
            for prediction in trace.policy_predictions
        ),
        "actions": tuple(
            (
                action.selected_chunk_index,
                action.requested_action.tobytes(),
                action.applied_action.tobytes(),
                action.modification_reason,
            )
            for action in trace.executed_actions
        ),
        "signals": tuple((signal.name, np.asarray(signal.value).tobytes()) for signal in trace.evaluation_signals),
        "metrics": tuple(
            (
                result.metric_id, result.metric_version, result.status.value, result.value,
                result.metric_config_hash,
            )
            for result in metrics
        ),
    }


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


def test_gate_a_configured_run_is_immutable_offline_reproducible_and_deterministic(tmp_path) -> None:
    resolved = ConfigResolver(CONFIGS, repository_root=REPOSITORY).resolve(
        "configs/experiments/mock-e2e-smoke.yaml", local_profile=_profile(tmp_path)
    )
    repeated_resolution = ConfigResolver(CONFIGS, repository_root=REPOSITORY).resolve(
        "configs/experiments/mock-e2e-smoke.yaml", local_profile=_profile(tmp_path)
    )
    assert resolved.scientific_config_hash == repeated_resolution.scientific_config_hash
    assert resolved.execution_config_hash == repeated_resolution.execution_config_hash

    registry = MetricRegistry.default()
    run_signatures = []
    run_ids = ("gate-a/mock-run-1", "gate-a/mock-run-2")
    for run_id in run_ids:
        plan = resolved.create_plan(
            make_run_context(run_id=run_id, seed=17),
            (TaskId("mock-task-0"), TaskId("mock-task-1")),
        )
        events = []
        benchmark = TrackingBenchmark(
            maximum_steps=resolved.benchmark_settings.maximum_episode_steps,
            modify_actions=resolved.benchmark_settings.modify_actions,
            task_count=resolved.benchmark_settings.task_count,
            terminal_outcomes=resolved.benchmark_settings.terminal_outcomes,
            event_log=events,
        )
        policy = TrackingPolicy(horizon=resolved.policy_settings.horizon, event_log=events)
        store = FilesystemRunArtifactStore(resolved.artifact_settings.root)
        runner = ExperimentRunner(
            plan, benchmark, policy, store, metric_registry=registry, clock=DeterministicClock(),
            provenance_provider=StaticProvenanceProvider(ProvenanceSnapshot(
                ovlab_git_commit="test-working-tree",
                ovlab_dirty=True,
                external_commits={"libero": LIBERO_COMMIT, "openvla": OPENVLA_COMMIT},
                environment_snapshot_reference="conda:ovlab-tester",
            )),
            configuration_snapshot=resolved.configuration_snapshot(),
        )
        report = runner.connect()
        task_results = runner.run()

        assert runner.state is RunnerState.COMPLETED
        assert report.compatibility_report.compatible
        assert tuple(str(task.task_id) for task in report.selected_tasks) == ("mock-task-0", "mock-task-1")
        assert benchmark.closed and policy.closed
        assert benchmark.reset_contexts == policy.reset_contexts
        expected_contexts = tuple(
            make_episode_context(plan, task, task_index, rollout_index, DeterministicClock())
            for task_index, task in enumerate(report.selected_tasks)
            for rollout_index in range(plan.rollout_count_per_task)
        )
        assert tuple(
            (context.episode_id, context.seed) for context in benchmark.reset_contexts
        ) == tuple((context.episode_id, context.seed) for context in expected_contexts)
        assert all(isinstance(observation, PolicyObservation) for observation in policy.observations)
        assert all(
            "hidden_target" not in {item.name for item in observation.images + observation.proprioception}
            for observation in policy.observations
        )
        assert events[0][0] == "benchmark.initialize"
        assert events[1][0] == "policy.initialize"
        assert events[-2:] == [("policy.close", None), ("benchmark.close", None)]
        for context in benchmark.reset_contexts:
            assert events.index(("policy.reset", str(context.episode_id))) < events.index(
                ("benchmark.reset", str(context.episode_id))
            )

        run_path = store._run_path(plan.run_context.run_id)
        started = json.loads((run_path / "manifest.started.json").read_text(encoding="utf-8"))
        completed = json.loads((run_path / "manifest.completed.json").read_text(encoding="utf-8"))
        connection = json.loads((run_path / "connection.json").read_text(encoding="utf-8"))
        source = load(run_path / "source_config.yaml")
        preserved = load(run_path / "resolved_config.yaml")
        assert started["scientific_config_hash"] == resolved.scientific_config_hash
        assert started["execution_config_hash"] == resolved.execution_config_hash
        assert started["provenance"]["external_commits"] == {
            "libero": LIBERO_COMMIT, "openvla": OPENVLA_COMMIT,
        }
        assert started["recording_policy_hash"] == plan.trace_recording_policy.hash
        assert connection["compatible"] is True
        assert source["kind"] == "scientific_experiment"
        assert "/home/" not in (run_path / "source_config.yaml").read_text(encoding="utf-8")
        assert preserved["scientific_config_hash"] == resolved.scientific_config_hash
        assert preserved["execution_config_hash"] == resolved.execution_config_hash
        assert completed["status"] == "completed"
        assert completed["episode_count"] == 4
        assert completed["episode_counts_by_terminal_status"] == {"success": 2, "time_limit": 2}

        traces_and_metrics = []
        by_task = {str(task.task_id): [] for task in report.selected_tasks}
        for context in benchmark.reset_contexts:
            trace = store.read_episode_trace(context.run_id, context.task_id, context.episode_id)
            stored_metrics = store.read_metric_results(context.run_id, context.task_id, context.episode_id)
            offline_metrics = _offline_episode_results(trace, plan, registry)
            assert offline_metrics == stored_metrics
            by_task[str(context.task_id)].extend(offline_metrics)
            assert trace.episode_context == context
            assert len(trace.executed_actions) == 3
            assert all(action.requested_action.tobytes() != action.applied_action.tobytes() for action in trace.executed_actions)
            assert all(action.metadata["closed_loop_step_duration_ns"] > 0 for action in trace.executed_actions)
            values = {result.metric_id: result for result in stored_metrics}
            expected_success = 1 if str(context.task_id) == "mock-task-0" else 0
            assert values["task.success"].value == expected_success
            assert values["failure.invalid_prediction_rate"].value == 0.0
            assert values["failure.action_modification_rate"].value == 1.0
            assert values["failure.repeated_no_op_rate"].value == 0.0
            assert values["failure.gripper_flicker_rate"].value == 0.0
            assert values["failure.collision_rate"].status is MetricStatus.UNAVAILABLE
            assert values["failure.collision_rate"].value is None
            assert values["system.inference_latency"].value == 1e-6
            applied = np.stack([action.applied_action for action in trace.executed_actions])
            assert values["action.variance"].value == pytest.approx(float(np.mean(np.var(applied, axis=0))))
            assert values["action.smoothness_1"].value == pytest.approx(
                float(np.mean(np.linalg.norm(np.diff(applied, axis=0), axis=1)))
            )
            second = applied[2:] - 2 * applied[1:-1] + applied[:-2]
            assert values["action.smoothness_2"].value == pytest.approx(
                float(np.mean(np.linalg.norm(second, axis=1)))
            )
            episode_path = store._episode_path(context.run_id, context.task_id, context.episode_id)
            assert not tuple(episode_path.rglob("*.pkl")) and not tuple(episode_path.rglob("*.pickle"))
            assert tuple(episode_path.rglob("*.npy"))
            traces_and_metrics.append((trace, stored_metrics))

        for task in report.selected_tasks:
            offline_task = _offline_task_results(task, by_task[str(task.task_id)], plan, registry)
            stored_task = store.read_metric_results(plan.run_context.run_id, task.task_id)
            assert offline_task == stored_task == task_results[str(task.task_id)]
            success_rate = next(item for item in stored_task if item.metric_id == "task.success_rate")
            assert success_rate.value == (1.0 if str(task.task_id) == "mock-task-0" else 0.0)

        first_trace = traces_and_metrics[0][0]
        with pytest.raises(ArtifactError, match="already exists"):
            store.write_episode_trace(plan.run_context.run_id, first_trace)
        preserved_trace = store.read_episode_trace(
            first_trace.episode_context.run_id,
            first_trace.episode_context.task_id,
            first_trace.episode_context.episode_id,
        )
        assert preserved_trace.episode_context == first_trace.episode_context
        assert tuple(action.applied_action.tobytes() for action in preserved_trace.executed_actions) == tuple(
            action.applied_action.tobytes() for action in first_trace.executed_actions
        )
        run_signatures.append(tuple(
            _scientific_signature(trace, metrics) for trace, metrics in traces_and_metrics
        ))

    assert run_signatures[0] == run_signatures[1]
