"""Explicit deterministic registry for built-in metric plug-ins."""

from .action.metrics import ActionVarianceMetric, Smoothness1Metric, Smoothness2Metric
from .errors import MetricRegistryError
from .failure.metrics import (
    ActionModificationRateMetric,
    CollisionRateMetric,
    GripperFlickerRateMetric,
    InvalidPredictionRateMetric,
    RepeatedNoOpRateMetric,
)
from .system.inference_latency import InferenceLatencyMetric
from .task.success import TaskSuccessMetric, TaskSuccessRateMetric


class MetricRegistry:
    def __init__(self, plugins=()) -> None:
        self._plugins = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin) -> None:
        descriptor = plugin.descriptor
        if descriptor.metric_id in self._plugins:
            existing = self._plugins[descriptor.metric_id].descriptor
            if existing.metric_version != descriptor.metric_version:
                raise MetricRegistryError(
                    f"conflicting versions for {descriptor.metric_id}: {existing.metric_version} and {descriptor.metric_version}"
                )
            raise MetricRegistryError(f"duplicate metric ID: {descriptor.metric_id}")
        self._plugins[descriptor.metric_id] = plugin

    def resolve(self, metric_id: str):
        try:
            return self._plugins[metric_id]
        except KeyError as exc:
            raise MetricRegistryError(f"unknown metric ID: {metric_id}") from exc

    def plugins(self):
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    def descriptors(self):
        return tuple(plugin.descriptor for plugin in self.plugins())

    @classmethod
    def default(cls):
        return cls(
            (
                ActionVarianceMetric(),
                Smoothness1Metric(),
                Smoothness2Metric(),
                ActionModificationRateMetric(),
                CollisionRateMetric(),
                GripperFlickerRateMetric(),
                InvalidPredictionRateMetric(),
                RepeatedNoOpRateMetric(),
                InferenceLatencyMetric(),
                TaskSuccessMetric(),
                TaskSuccessRateMetric(),
            )
        )
