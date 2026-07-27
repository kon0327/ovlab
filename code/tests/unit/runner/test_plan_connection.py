"""Experiment plan, seeds, connection preflight, and lifecycle tests."""

from dataclasses import replace

import pytest

from helpers.mock_policy import MockPolicy
from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_core.contracts import TaskId
from ovlab_metrics import MetricRegistry
from ovlab_runner import (
    ConnectionError, DeterministicClock, ExperimentRunner, InMemoryRunArtifactStore,
    MetricAvailabilityPolicy, RunnerError, RunnerLifecycleError, RunnerState,
)


class RuntimeMetadataBenchmark(TrackingBenchmark):
    def __init__(self):
        super().__init__()
        self.detected_renderer = None

    def runtime_metadata(self):
        return {
            "libero_renderer": {
                "requested_backend": "egl",
                "resolved_backend": "egl",
                "device_id": 0,
                "override_source": None,
                "detected_renderer": self.detected_renderer,
            }
        }

    def _step(self, request):
        result = super()._step(request)
        self.detected_renderer = {"renderer": "synthetic-egl"}
        return result


def test_plan_validation_hash_and_seed_schedule() -> None:
    plan = runner_plan()
    assert plan.hash == runner_plan().hash
    assert plan.episode_seed(TaskId("mock-task-0"), 0, 0) == runner_plan().episode_seed(TaskId("mock-task-0"), 0, 0)
    assert plan.episode_seed(TaskId("mock-task-0"), 0, 0) != plan.episode_seed(TaskId("mock-task-0"), 0, 1)
    with pytest.raises(RunnerError): runner_plan(selected_task_ids=(TaskId("x"), TaskId("x")))
    with pytest.raises(RunnerError): runner_plan(enabled_metric_ids=("x", "x"))


def test_required_statically_unavailable_metric_blocks_but_optional_is_allowed() -> None:
    optional = runner_plan(enabled_metric_ids=("failure.collision_rate",))
    runner = ExperimentRunner(optional, TrackingBenchmark(), TrackingPolicy(), InMemoryRunArtifactStore(), clock=DeterministicClock())
    report = runner.connect()
    assert report.potentially_unavailable_metric_ids == ("failure.collision_rate",)
    runner.close()
    required = runner_plan(
        enabled_metric_ids=("failure.collision_rate",),
        required_metric_ids=("failure.collision_rate",),
        unavailable_metric_policy=MetricAvailabilityPolicy.REQUIRE_SELECTED,
    )
    with pytest.raises(ConnectionError):
        ExperimentRunner(required, TrackingBenchmark(), TrackingPolicy(), InMemoryRunArtifactStore()).connect()


def test_incompatible_capabilities_block_connection() -> None:
    policy = MockPolicy()
    capabilities = policy._initialize(runner_plan().run_context)
    incompatible = replace(capabilities, contract_version="different")
    with pytest.raises(ConnectionError):
        ExperimentRunner(
            runner_plan(), TrackingBenchmark(), TrackingPolicy(capabilities_override=incompatible),
            InMemoryRunArtifactStore(),
        ).connect()


def test_runner_lifecycle_and_close_are_explicit() -> None:
    benchmark, policy = TrackingBenchmark(), TrackingPolicy()
    runner = ExperimentRunner(runner_plan(), benchmark, policy, InMemoryRunArtifactStore())
    with pytest.raises(RunnerLifecycleError): runner.run()
    runner.connect()
    with pytest.raises(RunnerLifecycleError): runner.connect()
    runner.close()
    runner.close()
    assert runner.state is RunnerState.CLOSED and benchmark.closed and policy.closed


def test_renderer_runtime_metadata_is_recorded_before_and_after_execution() -> None:
    store = InMemoryRunArtifactStore()
    runner = ExperimentRunner(
        runner_plan(),
        RuntimeMetadataBenchmark(),
        TrackingPolicy(),
        store,
        clock=DeterministicClock(),
    )
    runner.connect()
    started = store.runs["runner-test"]["started"]["runtime"]["benchmark"]["libero_renderer"]
    assert started == {
        "requested_backend": "egl",
        "resolved_backend": "egl",
        "device_id": 0,
        "override_source": None,
        "detected_renderer": None,
    }

    runner.run()
    completed = store.runs["runner-test"]["completed"]["metadata"]["benchmark_runtime"]["libero_renderer"]
    assert completed["requested_backend"] == "egl"
    assert completed["resolved_backend"] == "egl"
    assert completed["device_id"] == 0
    assert completed["detected_renderer"] == {"renderer": "synthetic-egl"}
