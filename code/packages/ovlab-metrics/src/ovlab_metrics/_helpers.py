"""Internal helpers shared by built-in metrics."""

from .config import canonical_config, config_hash
from .descriptor import MetricScope
from .results import MetricResult, MetricStatus


def episode_result(plugin, trace, config, status, value=None, *, samples=0, reason=None, diagnostics=None, metadata=None):
    context = trace.episode_context
    return MetricResult(
        plugin.descriptor.metric_id,
        plugin.descriptor.metric_version,
        MetricScope.EPISODE,
        status,
        value,
        plugin.descriptor.result_unit,
        samples,
        context.run_id,
        context.task_id,
        context.episode_id,
        reason,
        diagnostics or {},
        canonical_config(config),
        config_hash(config),
        metadata or {},
    )


def available(plugin, trace, config, value, *, samples, diagnostics=None, metadata=None):
    return episode_result(
        plugin, trace, config, MetricStatus.AVAILABLE, value, samples=samples, diagnostics=diagnostics, metadata=metadata
    )
