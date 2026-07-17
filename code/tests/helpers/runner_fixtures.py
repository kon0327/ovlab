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
    def __init__(self, *, event_log=None, **kwargs):
        super().__init__(**kwargs)
        self.closed = False
        self.reset_contexts = []
        self.event_log = [] if event_log is None else event_log

    def _initialize(self, context):
        self.event_log.append(("benchmark.initialize", str(context.run_id)))
        return super()._initialize(context)

    def _list_tasks(self):
        self.event_log.append(("benchmark.list_tasks", None))
        return super()._list_tasks()

    def _reset_episode(self, context):
        self.event_log.append(("benchmark.reset", str(context.episode_id)))
        self.reset_contexts.append(context)
        return super()._reset_episode(context)

    def _step(self, request):
        self.event_log.append(("benchmark.step", str(request.step_context.step_id)))
        return super()._step(request)

    def _close(self):
        self.event_log.append(("benchmark.close", None))
        self.closed = True


class TrackingPolicy(MockPolicy):
    def __init__(self, *, event_log=None, **kwargs):
        super().__init__(**kwargs)
        self.closed = False
        self.reset_contexts = []
        self.observations = []
        self.event_log = [] if event_log is None else event_log

    def _initialize(self, context):
        self.event_log.append(("policy.initialize", str(context.run_id)))
        return super()._initialize(context)

    def _reset_episode(self, context):
        self.event_log.append(("policy.reset", str(context.episode_id)))
        self.reset_contexts.append(context)
        return super()._reset_episode(context)

    def _predict(self, observation):
        self.event_log.append(("policy.predict", str(observation.step_id)))
        self.observations.append(observation)
        return super()._predict(observation)

    def _end_episode(self, context):
        self.event_log.append(("policy.end", str(context.episode_id)))

    def _close(self):
        self.event_log.append(("policy.close", None))
        self.closed = True
