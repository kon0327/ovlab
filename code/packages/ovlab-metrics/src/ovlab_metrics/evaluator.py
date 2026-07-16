"""Requirement-aware deterministic offline episode evaluator."""

from ._helpers import episode_result
from .errors import ActionSequenceError
from .plugin import EpisodeMetricPlugin
from .requirements import resolve_requirements
from .results import MetricStatus


class MetricEvaluator:
    def __init__(self, registry, *, strict: bool = False) -> None:
        self.registry = registry
        self.strict = strict

    def evaluate(self, trace, configs=None):
        configs = configs or {}
        results = []
        for plugin in self.registry.plugins():
            if not isinstance(plugin, EpisodeMetricPlugin):
                continue
            config = configs.get(plugin.descriptor.metric_id, plugin.default_config)
            resolution = resolve_requirements(trace, plugin.requirements)
            if resolution.missing_requirements or resolution.incompatible_requirements:
                reason = "; ".join(resolution.missing_requirements + resolution.incompatible_requirements)
                results.append(
                    episode_result(
                        plugin, trace, config, MetricStatus.UNAVAILABLE, reason=reason,
                        diagnostics={"warnings": resolution.warnings},
                    )
                )
                continue
            if resolution.insufficient_requirements:
                results.append(
                    episode_result(
                        plugin, trace, config, MetricStatus.INSUFFICIENT_DATA,
                        reason="; ".join(resolution.insufficient_requirements),
                    )
                )
                continue
            try:
                result = plugin.evaluate(trace, config)
            except ActionSequenceError as exc:
                result = episode_result(plugin, trace, config, MetricStatus.UNAVAILABLE, reason=str(exc))
            except Exception as exc:
                if self.strict:
                    raise
                result = episode_result(
                    plugin, trace, config, MetricStatus.ERROR,
                    reason=f"{type(exc).__name__}: metric evaluation failed",
                )
            results.append(result)
        return tuple(sorted(results, key=lambda result: (result.metric_id, result.metric_version)))
