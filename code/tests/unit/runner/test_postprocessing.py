"""Reporting lifecycle isolation regressions that remain dependency-light."""

import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_runner import (
    AutomaticDerivedReporter, DeterministicClock, ExperimentRunner,
    InMemoryRunArtifactStore, RunnerState,
)


class BrokenReporter:
    def __init__(self):
        self.calls = []
        self.failures = []

    def task_finalized(self, run_id, task_id):
        self.calls.append(("task", str(task_id)))
        raise RuntimeError("renderer exploded")

    def run_finalized(self, run_id, status):
        self.calls.append(("run", status))
        raise RuntimeError("renderer exploded")

    def record_failure(self, run_id, scope, exc):
        self.failures.append((scope, type(exc).__name__))


def test_reporting_failure_cannot_reclassify_or_interrupt_completed_run():
    store = InMemoryRunArtifactStore()
    reporter = BrokenReporter()
    runner = ExperimentRunner(
        runner_plan(), TrackingBenchmark(maximum_steps=3), TrackingPolicy(), store,
        clock=DeterministicClock(), postprocessor=reporter, postprocessor_failure_policy="warn",
    )
    runner.connect()
    with pytest.warns(RuntimeWarning, match="without changing canonical benchmark status"):
        runner.run()
    assert runner.state is RunnerState.COMPLETED
    assert store.runs["runner-test"]["completed"]["status"] == "completed"
    assert reporter.calls == [("task", "mock-task-0"), ("run", "completed")]
    assert reporter.failures == [("task", "RuntimeError"), ("run", "RuntimeError")]
    assert store.write_order.index("task-metrics:mock-task-0") < store.write_order.index("manifest.completed")


def test_automatic_reporting_cannot_be_configured_to_reclassify_a_run():
    with pytest.raises(ValueError, match="failure_policy"):
        ExperimentRunner(
            runner_plan(), TrackingBenchmark(), TrackingPolicy(), InMemoryRunArtifactStore(),
            postprocessor_failure_policy="raise",
        )


def test_completed_run_automatically_generates_isolated_export_only_at_run_finalization():
    class Reporter:
        def __init__(self):
            self.generated = []

        def generate(self, run_id):
            self.generated.append(run_id)

        def record_failure(self, run_id, scope, exc):
            raise AssertionError("unexpected failure")

    class Exporter:
        def __init__(self):
            self.generated = []

        def generate_isolated(self, run_id):
            self.generated.append(run_id)

    reporter, exporter = Reporter(), Exporter()
    postprocessor = AutomaticDerivedReporter(
        reporter, on_task_finalize=True, on_run_finalize=True,
        isolated_export_engine=exporter,
    )
    postprocessor.task_finalized("run-1", "task-1")
    assert reporter.generated == ["run-1"]
    assert exporter.generated == []
    postprocessor.run_finalized("run-1", "completed")
    assert reporter.generated == ["run-1", "run-1"]
    assert exporter.generated == ["run-1"]
    postprocessor.run_finalized("run-2", "failed")
    assert exporter.generated == ["run-1"]

    export_only = Exporter()
    reporting_disabled = AutomaticDerivedReporter(None, isolated_export_engine=export_only)
    reporting_disabled.task_finalized("run-3", "task-1")
    reporting_disabled.run_finalized("run-3", "completed")
    assert export_only.generated == ["run-3"]
