"""Authoritative episode success and task success-rate metrics."""

from ovlab_core.contracts import EpisodeTerminalStatus, SignalAccess

from .._helpers import available, episode_result
from ..config import EmptyMetricConfig, SuccessRateMetricConfig, canonical_config, config_hash
from ..descriptor import MetricDescriptor, MetricLevel, MetricScope, OptimizationDirection
from ..errors import MetricAggregationError
from ..plugin import EpisodeMetricPlugin, TaskMetricPlugin
from ..requirements import MetricRequirements, SignalRequirement, TraceField, TraceFieldRequirement
from ..results import MetricResult, MetricStatus


class TaskSuccessMetric(EpisodeMetricPlugin):
    descriptor = MetricDescriptor(
        "task.success", "Task success", "Authoritative benchmark success", "1.0.0",
        MetricLevel.TASK, False, "boolean", OptimizationDirection.HIGHER, (MetricScope.EPISODE,),
    )
    requirements = MetricRequirements(
        (TraceFieldRequirement(TraceField.TERMINAL_STATUS),),
        (SignalRequirement("benchmark.task_success", "bool", (), (SignalAccess.EVALUATION_ONLY,)),),
    )
    default_config = EmptyMetricConfig()

    def evaluate(self, trace, config):
        signals = [signal for signal in trace.evaluation_signals if signal.name == "benchmark.task_success"]
        if not signals:
            return episode_result(
                self, trace, config, MetricStatus.UNAVAILABLE, reason="authoritative success signal missing",
                diagnostics={"terminal_status": trace.terminal_status.value},
            )
        success = bool(signals[-1].value)
        status_success = trace.terminal_status is EpisodeTerminalStatus.SUCCESS
        contradictory = success != status_success
        if contradictory:
            return episode_result(
                self, trace, config, MetricStatus.ERROR, reason="success signal contradicts terminal status",
                diagnostics={"terminal_status": trace.terminal_status.value, "signal": success},
            )
        return available(
            self, trace, config, int(success), samples=1,
            diagnostics={"terminal_status": trace.terminal_status.value},
        )


class TaskSuccessRateMetric(TaskMetricPlugin):
    descriptor = MetricDescriptor(
        "task.success_rate", "Task success rate", "Macro task success rate", "1.0.0",
        MetricLevel.TASK, False, "ratio", OptimizationDirection.HIGHER, (MetricScope.TASK,),
    )
    default_config = SuccessRateMetricConfig()

    def aggregate(self, task_context, episode_results, config):
        results = tuple(episode_results)
        if not results:
            raise MetricAggregationError("success rate requires episode results")
        successes = eligible = excluded = unavailable = 0
        for result in results:
            if result.status is MetricStatus.ERROR:
                raise MetricAggregationError("error success results must not be silently dropped")
            terminal = result.diagnostics.get("terminal_status")
            if terminal == EpisodeTerminalStatus.BENCHMARK_ERROR.value:
                excluded += 1
                continue
            if terminal == EpisodeTerminalStatus.ABORTED.value and not config.aborted_policy_attributable:
                excluded += 1
                continue
            if result.status is MetricStatus.AVAILABLE:
                eligible += 1
                successes += bool(result.value)
            else:
                unavailable += 1
        if eligible == 0:
            status, value, reason = MetricStatus.UNAVAILABLE, None, "no eligible success results"
        else:
            status, value, reason = MetricStatus.AVAILABLE, successes / eligible, None
        first = results[0]
        return MetricResult(
            self.descriptor.metric_id, self.descriptor.metric_version, MetricScope.TASK, status, value,
            self.descriptor.result_unit, eligible, task_context.run_id, task_context.task_id, None, reason,
            {
                "numerator": successes,
                "denominator": eligible,
                "excluded_episode_count": excluded,
                "unavailable_episode_count": unavailable,
                "aborted_policy_attributable": config.aborted_policy_attributable,
            },
            canonical_config(config), config_hash(config),
        )
