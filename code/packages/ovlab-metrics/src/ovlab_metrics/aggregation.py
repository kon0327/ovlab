"""Safe macro aggregation of homogeneous episode metric results."""

import json

from .descriptor import MetricScope
from .errors import MetricAggregationError
from .results import MetricResult, MetricStatus, MetricSummary


def aggregate_episode_results(task_context, episode_results) -> MetricResult:
    results = tuple(episode_results)
    if not results:
        raise MetricAggregationError("aggregation requires episode results")
    first = results[0]
    identity = (
        first.run_id,
        first.task_id,
        first.metric_id,
        first.metric_version,
        first.metric_config_hash,
        first.unit,
        first.metadata.get("action_source"),
        first.metadata.get("action_spec"),
    )
    for result in results[1:]:
        candidate = (
            result.run_id,
            result.task_id,
            result.metric_id,
            result.metric_version,
            result.metric_config_hash,
            result.unit,
            result.metadata.get("action_source"),
            result.metadata.get("action_spec"),
        )
        if candidate != identity:
            raise MetricAggregationError("episode results differ in identity, configuration, source, or action spec")
    if task_context.run_id != first.run_id or task_context.task_id != first.task_id:
        raise MetricAggregationError("task context does not match episode results")
    errors = [result for result in results if result.status is MetricStatus.ERROR]
    if errors:
        raise MetricAggregationError("error results must be resolved explicitly before aggregation")
    values = [float(result.value) for result in results if result.status is MetricStatus.AVAILABLE]
    unavailable = sum(
        result.status in (MetricStatus.UNAVAILABLE, MetricStatus.INSUFFICIENT_DATA) for result in results
    )
    excluded = sum(result.status is MetricStatus.NOT_APPLICABLE for result in results)
    if not values:
        return MetricResult(
            first.metric_id, first.metric_version, MetricScope.TASK, MetricStatus.UNAVAILABLE, None,
            first.unit, 0, first.run_id, first.task_id, None, "no available episode values",
            {"unavailable_episode_count": unavailable, "excluded_episode_count": excluded},
            first.metric_config, first.metric_config_hash, first.metadata,
        )
    summary = MetricSummary.from_values(values, unavailable, excluded)
    return MetricResult(
        first.metric_id, first.metric_version, MetricScope.TASK, MetricStatus.AVAILABLE, summary.as_dict(),
        first.unit, len(values), first.run_id, first.task_id, None, None, summary.as_dict(),
        first.metric_config, first.metric_config_hash, first.metadata,
    )


def aggregate_results_by_task(task_contexts, episode_results) -> tuple[MetricResult, ...]:
    """Group by run, task, metric version, and configuration before macro aggregation."""
    contexts = {(context.run_id, context.task_id): context for context in task_contexts}
    groups = {}
    for result in episode_results:
        if result.episode_id is None:
            raise MetricAggregationError("grouping accepts episode-scoped results only")
        action_identity = json.dumps(
            {
                "source": result.metadata.get("action_source"),
                "spec": _plain(result.metadata.get("action_spec")),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = (
            result.run_id,
            result.task_id,
            result.metric_id,
            result.metric_version,
            result.metric_config_hash,
            action_identity,
        )
        groups.setdefault(key, []).append(result)
    aggregated = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        context_key = key[:2]
        if context_key not in contexts:
            raise MetricAggregationError(f"missing TaskContext for run/task {context_key}")
        aggregated.append(aggregate_episode_results(contexts[context_key], groups[key]))
    return tuple(aggregated)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
