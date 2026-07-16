"""Abstract episode and task metric plug-in interfaces."""

from abc import ABC, abstractmethod

from .descriptor import MetricDescriptor
from .requirements import MetricRequirements


class EpisodeMetricPlugin(ABC):
    descriptor: MetricDescriptor
    requirements: MetricRequirements = MetricRequirements()
    default_config = None

    @abstractmethod
    def evaluate(self, trace, config): ...


class TaskMetricPlugin(ABC):
    descriptor: MetricDescriptor
    requirements: MetricRequirements = MetricRequirements()
    default_config = None

    @abstractmethod
    def aggregate(self, task_context, episode_results, config): ...
