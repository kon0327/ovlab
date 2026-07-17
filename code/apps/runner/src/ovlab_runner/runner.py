"""In-process synchronous ExperimentRunner."""

from collections import Counter

from ovlab_metrics import (
    MetricEvaluator, MetricRegistry, MetricStatus, TaskMetricPlugin, aggregate_episode_results,
)

from .connection import connect_components
from .configuration import RunConfigurationSnapshot
from .errors import ConnectionError, ExperimentExecutionError, RunnerLifecycleError
from .execution import execute_episode
from .lifecycle import RunnerState, SystemClock
from .plan import EpisodeErrorPolicy
from .provenance import StaticProvenanceProvider


class ExperimentRunner:
    version = "0.1.0"

    def __init__(self, plan, benchmark, policy, artifact_store, *, metric_registry=None, clock=None, provenance_provider=None, configuration_snapshot=None):
        self.plan = plan
        self.benchmark = benchmark
        self.policy = policy
        self.store = artifact_store
        self.metric_registry = metric_registry or MetricRegistry.default()
        self.clock = clock or SystemClock()
        self.provenance_provider = provenance_provider or StaticProvenanceProvider()
        if configuration_snapshot is not None and not isinstance(configuration_snapshot, RunConfigurationSnapshot):
            raise TypeError("configuration_snapshot must be a RunConfigurationSnapshot or None")
        self.configuration_snapshot = configuration_snapshot
        self.state = RunnerState.CREATED
        self.connection_report = None
        self._resources_closed = False

    def connect(self):
        self._require(RunnerState.CREATED)
        try:
            report = connect_components(self.plan, self.benchmark, self.policy, self.metric_registry)
        except Exception as exc:
            self.state = RunnerState.FAILED
            self._close_resources()
            raise ConnectionError("runner connection failed") from exc
        self.connection_report = report
        started = {
            "run_id": str(self.plan.run_context.run_id), "plan_hash": self.plan.hash,
            "contract_version": self.plan.run_context.contract_version, "runner_version": self.version,
            "benchmark": {"name": report.benchmark_name, "version": report.benchmark_version},
            "policy": {"name": report.policy_name, "version": report.policy_version},
            "metrics": [{"id": item.metric_id, "version": item.metric_version} for item in report.enabled_metric_descriptors],
            "recording_policy_hash": self.plan.trace_recording_policy.hash,
            "start_wall_time_utc_ns": self.clock.wall_time_utc_ns(),
            "provenance": self.provenance_provider.collect().as_dict(), "metadata": dict(self.plan.metadata),
        }
        if self.configuration_snapshot is not None:
            started["scientific_config_hash"] = self.configuration_snapshot.scientific_config_hash
            started["execution_config_hash"] = self.configuration_snapshot.execution_config_hash
        try:
            self.store.create_run(self.plan.run_context.run_id, started)
            if self.configuration_snapshot is not None:
                self.store.write_configuration(self.plan.run_context.run_id, self.configuration_snapshot)
            self.store.write_plan(self.plan.run_context.run_id, self.plan)
            self.store.write_connection_report(self.plan.run_context.run_id, report)
        except Exception as exc:
            self.state = RunnerState.FAILED
            self._close_resources()
            raise ConnectionError("runner artifact initialization failed") from exc
        self.state = RunnerState.CONNECTED
        return report

    def run(self):
        self._require(RunnerState.CONNECTED)
        self.state = RunnerState.RUNNING
        terminal_counts = Counter()
        metric_errors = 0
        episode_count = 0
        all_task_results = {}
        try:
            for task_index, task in enumerate(self.connection_report.selected_tasks):
                episode_results = []
                stop_task = False
                for rollout_index in range(self.plan.rollout_count_per_task):
                    trace, error = execute_episode(
                        self.plan, task, task_index, rollout_index, self.benchmark, self.policy, self.clock
                    )
                    self.store.write_episode_trace(self.plan.run_context.run_id, trace)
                    terminal_counts[trace.terminal_status.value] += 1
                    episode_count += 1
                    results = self._evaluate(trace)
                    metric_errors += sum(result.status is MetricStatus.ERROR for result in results)
                    self.store.write_episode_metric_results(
                        self.plan.run_context.run_id, task.task_id, trace.episode_context.episode_id, results
                    )
                    episode_results.extend(results)
                    required_errors = [result for result in results if result.metric_id in self.plan.required_metric_ids and result.status is MetricStatus.ERROR]
                    if error is not None or required_errors:
                        if isinstance(error, KeyboardInterrupt): raise error
                        if self.plan.episode_error_policy is EpisodeErrorPolicy.STOP_RUN:
                            raise ExperimentExecutionError("episode failed under STOP_RUN policy") from error
                        if self.plan.episode_error_policy is EpisodeErrorPolicy.CONTINUE_RUN:
                            stop_task = True
                            break
                task_results = self._aggregate_task(task, episode_results)
                self.store.write_task_metric_results(self.plan.run_context.run_id, task.task_id, task_results)
                all_task_results[str(task.task_id)] = task_results
                if stop_task:
                    continue
            completed = self._final_manifest("completed", terminal_counts, episode_count, metric_errors)
            self.store.finalize_run(self.plan.run_context.run_id, completed)
            self.state = RunnerState.COMPLETED
            return all_task_results
        except BaseException as exc:
            self.state = RunnerState.FAILED
            failed = self._final_manifest("failed", terminal_counts, episode_count, metric_errors, type(exc).__name__)
            try: self.store.mark_run_failed(self.plan.run_context.run_id, failed)
            except Exception: pass
            raise
        finally:
            self._close_resources()

    def close(self):
        if self.state is RunnerState.CLOSED: return
        self._close_resources()
        self.state = RunnerState.CLOSED

    def _evaluate(self, trace):
        plugins = []
        for metric_id in self.plan.enabled_metric_ids:
            plugin = self.metric_registry.resolve(metric_id)
            if not isinstance(plugin, TaskMetricPlugin): plugins.append(plugin)
        if "task.success_rate" in self.plan.enabled_metric_ids and "task.success" not in self.plan.enabled_metric_ids:
            plugins.append(self.metric_registry.resolve("task.success"))
        registry = MetricRegistry(plugins)
        configs = {key: value for key, value in self.plan.metric_configurations.items() if key in {p.descriptor.metric_id for p in plugins}}
        return MetricEvaluator(registry).evaluate(trace, configs)

    def _aggregate_task(self, task, episode_results):
        aggregated = []
        for metric_id in self.plan.enabled_metric_ids:
            plugin = self.metric_registry.resolve(metric_id)
            if isinstance(plugin, TaskMetricPlugin):
                source_id = "task.success" if metric_id == "task.success_rate" else metric_id
                source = [result for result in episode_results if result.metric_id == source_id]
                config = self.plan.metric_configurations.get(metric_id, plugin.default_config)
                if source: aggregated.append(plugin.aggregate(_task_context(self.plan, task), source, config))
            else:
                source = [result for result in episode_results if result.metric_id == metric_id]
                if source:
                    try: aggregated.append(aggregate_episode_results(_task_context(self.plan, task), source))
                    except Exception:
                        if metric_id in self.plan.required_metric_ids: raise
        return tuple(aggregated)

    def _final_manifest(self, status, terminal_counts, episodes, metric_errors, failure_type=None):
        return {"status": status, "end_wall_time_utc_ns": self.clock.wall_time_utc_ns(), "task_count": len(self.connection_report.selected_tasks), "episode_count": episodes, "episode_counts_by_terminal_status": dict(sorted(terminal_counts.items())), "metric_error_count": metric_errors, "artifact_schema_version": "1.0.0", "failure_type": failure_type, "metadata": {}}

    def _close_resources(self):
        if self._resources_closed: return
        try: self.policy.close()
        finally: self.benchmark.close()
        self._resources_closed = True

    def _require(self, state):
        if self.state is not state: raise RunnerLifecycleError(f"operation requires {state.value}, current state is {self.state.value}")


def _task_context(plan, task):
    from ovlab_core.contracts import TaskContext
    return TaskContext(plan.run_context.run_id, task.task_id, task.suite_name, task.task_name, task.task_index, task.metadata)
