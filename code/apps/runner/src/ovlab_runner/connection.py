"""Benchmark-policy connection and metric preflight."""

from dataclasses import dataclass, field

from ovlab_core import CompatibilityReport, negotiate_capabilities
from ovlab_core.contracts import Metadata, normalize_metadata
from ovlab_metrics import EpisodeMetricPlugin, SignalRequirement, TraceField

from .errors import ConnectionError
from .plan import MetricAvailabilityPolicy


@dataclass(frozen=True, slots=True)
class ConnectionReport:
    benchmark_name: str
    benchmark_version: str
    policy_name: str
    policy_version: str
    contract_version: str
    compatibility_report: CompatibilityReport
    selected_tasks: tuple
    enabled_metric_descriptors: tuple
    required_metric_ids: tuple[str, ...]
    statically_available_metric_ids: tuple[str, ...]
    potentially_unavailable_metric_ids: tuple[str, ...]
    plan_hash: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "selected_tasks", tuple(self.selected_tasks))
        object.__setattr__(self, "enabled_metric_descriptors", tuple(self.enabled_metric_descriptors))
        object.__setattr__(self, "required_metric_ids", tuple(self.required_metric_ids))
        object.__setattr__(self, "statically_available_metric_ids", tuple(self.statically_available_metric_ids))
        object.__setattr__(self, "potentially_unavailable_metric_ids", tuple(self.potentially_unavailable_metric_ids))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))


def connect_components(plan, benchmark, policy, metric_registry) -> ConnectionReport:
    benchmark_caps = benchmark.initialize(plan.run_context)
    policy_caps = policy.initialize(plan.run_context)
    compatibility = negotiate_capabilities(benchmark_caps, policy_caps)
    compatibility.require_compatible()
    tasks_by_id = {task.task_id: task for task in benchmark.list_tasks()}
    missing_tasks = [str(task_id) for task_id in plan.selected_task_ids if task_id not in tasks_by_id]
    if missing_tasks:
        raise ConnectionError(f"selected benchmark tasks do not exist: {', '.join(missing_tasks)}")
    selected = tuple(tasks_by_id[task_id] for task_id in plan.selected_task_ids)
    plugins = []
    for metric_id in plan.enabled_metric_ids:
        try:
            plugins.append(metric_registry.resolve(metric_id))
        except Exception as exc:
            raise ConnectionError(f"enabled metric is not registered: {metric_id}") from exc
    available, conditional = [], []
    signal_specs = {spec.name: spec for spec in benchmark_caps.signal_registry}
    recording = plan.trace_recording_policy
    for plugin in plugins:
        static_missing = []
        for requirement in plugin.requirements.signals:
            spec = signal_specs.get(requirement.name)
            if spec is None and requirement.required:
                static_missing.append(f"signal:{requirement.name}")
            elif spec is not None:
                if spec.access.value == "privileged" and not recording.record_privileged_signals:
                    static_missing.append(f"recording-disabled:{requirement.name}")
                if spec.access.value == "evaluation_only" and not recording.record_evaluation_signals:
                    static_missing.append(f"recording-disabled:{requirement.name}")
        if static_missing:
            conditional.append(plugin.descriptor.metric_id)
        else:
            available.append(plugin.descriptor.metric_id)
    blocked = sorted(set(plan.required_metric_ids) & set(conditional))
    if blocked and plan.unavailable_metric_policy is MetricAvailabilityPolicy.REQUIRE_SELECTED:
        raise ConnectionError(f"required metrics are statically unavailable: {', '.join(blocked)}")
    return ConnectionReport(
        benchmark_caps.component_name,
        benchmark_caps.component_version,
        policy_caps.component_name,
        policy_caps.component_version,
        benchmark_caps.contract_version,
        compatibility,
        selected,
        tuple(plugin.descriptor for plugin in plugins),
        plan.required_metric_ids,
        tuple(sorted(available)),
        tuple(sorted(conditional)),
        plan.hash,
        {"metric_runtime_samples_are_conditional": True},
    )
