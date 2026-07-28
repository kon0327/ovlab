"""Offline metric recomputation from finalized immutable inference traces."""

from __future__ import annotations

import json
from pathlib import Path

from ovlab_metrics import EpisodeMetricPlugin, MetricEvaluator, MetricRegistry, config_from_canonical

from .artifacts.codec import TraceCodec
from .artifacts.filesystem import _decode_metric, _metric
from .inspection import RunIntegrityError, verify_run


class MetricRecomputationError(RuntimeError):
    """Stored trace or metric data cannot be recomputed consistently."""


def recompute_run_metrics(value: str | Path, *, registry: MetricRegistry | None = None) -> dict[str, object]:
    """Recompute episode metrics without policy, simulator, or trace mutation."""
    path = Path(value).expanduser().resolve()
    verify_run(path)
    try:
        plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
        enabled = tuple(plan["enabled_metric_ids"])
        recorded_configs = plan.get("metric_configurations", {})
        configurations = {
            metric_id: config_from_canonical(metric_id, recorded_configs[metric_id])
            for metric_id in recorded_configs
        }
    except Exception as exc:
        raise MetricRecomputationError("stored plan has invalid metric configuration") from exc
    source_registry = registry or MetricRegistry.default()
    try:
        plugins = tuple(
            source_registry.resolve(metric_id)
            for metric_id in enabled
            if isinstance(source_registry.resolve(metric_id), EpisodeMetricPlugin)
        )
    except Exception as exc:
        raise MetricRecomputationError("stored plan references an unavailable metric implementation") from exc
    evaluator = MetricEvaluator(MetricRegistry(plugins))
    codec = TraceCodec()
    comparisons = []
    all_agree = True
    for episode in sorted(path.glob("tasks/*/episodes/*")):
        try:
            trace = codec.decode(episode)
            recorded = tuple(
                _decode_metric(item)
                for item in json.loads((episode / "metrics.episode.json").read_text(encoding="utf-8"))
            )
            recomputed = evaluator.evaluate(
                trace,
                {key: item for key, item in configurations.items() if key in {p.descriptor.metric_id for p in plugins}},
            )
        except Exception as exc:
            raise MetricRecomputationError(f"metric recomputation failed for {episode.relative_to(path)}") from exc
        agree = recomputed == recorded
        all_agree = all_agree and agree
        comparisons.append({
            "episode_path": str(episode.relative_to(path)),
            "agree": agree,
            "recorded": [_metric(item) for item in recorded],
            "recomputed": [_metric(item) for item in recomputed],
        })
    return {
        "metric_api": "ovlab-metrics/offline@0.1.0",
        "trace_source": ".",
        "original_trace_modified": False,
        "all_results_agree": all_agree,
        "episode_count": len(comparisons),
        "comparisons": comparisons,
    }
