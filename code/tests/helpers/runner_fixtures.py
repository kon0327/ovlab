"""Runner plans and instrumented mock adapters."""

from dataclasses import replace

from helpers.contexts import make_run_context
from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from ovlab_core.contracts import TaskId
from ovlab_runner import (
    ActionExecutionPolicy, ArtifactStoreSettings, DeterministicClock, ExperimentPlan,
    TraceRecordingPolicy,
)


def runner_plan(**overrides):
    values = {
        "run_context": make_run_context(run_id="runner-test", seed=5),
        "selected_task_ids": (TaskId("mock-task-0"),),
        "rollout_count_per_task": 1,
        "base_episode_seed": 17,
        "default_maximum_episode_steps": 3,
        "action_execution_policy": ActionExecutionPolicy(),
        "enabled_metric_ids": ("action.variance", "system.inference_latency"),
        "metric_configurations": {},
        "required_metric_ids": (),
        "trace_recording_policy": TraceRecordingPolicy(record_privileged_signals=False),
        "artifact_store_settings": ArtifactStoreSettings("runs"),
    }
    values.update(overrides)
    return ExperimentPlan(**values)


class TrackingBenchmark(MockBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed = False
        self.reset_contexts = []

    def _reset_episode(self, context):
        self.reset_contexts.append(context)
        return super()._reset_episode(context)

    def _close(self): self.closed = True


class TrackingPolicy(MockPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed = False
        self.reset_contexts = []

    def _reset_episode(self, context):
        self.reset_contexts.append(context)
        return super()._reset_episode(context)

    def _close(self): self.closed = True
