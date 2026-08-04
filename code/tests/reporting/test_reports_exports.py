"""Gate H.2 focused report, export, identity, and integrity qualification."""

from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

from helpers.contexts import make_run_context
from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_benchctl.application import OvlabApplication
from ovlab_metrics import ActionSequenceMetricConfig, ActionSource
from ovlab_runner import (
    ArtifactError, AutomaticDerivedReporter, DerivedReportEngine, DeterministicClock,
    ExperimentRunner, ExportEngine, FilesystemRunArtifactStore, ReportProfile,
    RunConfigurationSnapshot, build_report_model, builtin_profile, validate_export_spec,
)
from ovlab_runner.derived import _trace_view


REPOSITORY = Path(__file__).resolve().parents[3]


def test_trace_report_extracts_policy_allocator_vram_and_estimated_compute(tmp_path):
    performance = MappingProxyType({
        "schema_version": "ovlab.performance-telemetry/v1",
        "cuda_memory_after": MappingProxyType({
            "status": "available", "allocated_bytes": 1024 * 1024,
            "reserved_bytes": 2 * 1024 * 1024,
            "peak_allocated_bytes": 3 * 1024 * 1024,
            "peak_reserved_bytes": 4 * 1024 * 1024,
        }),
        "estimated_compute": MappingProxyType({
            "status": "available", "estimated_gflops": 12.5,
            "method": "test-estimator", "formula": "test", "qualification": "estimated",
        }),
    })
    trace = SimpleNamespace(
        executed_actions=(SimpleNamespace(applied_action=[0.0] * 7, metadata={}),),
        policy_predictions=(SimpleNamespace(
            inference_duration_ns=1_000_000,
            metadata=MappingProxyType({"runtime": MappingProxyType({"performance": performance})}),
        ),),
    )
    view = _trace_view(tmp_path / "episode", trace)
    assert view["vram_allocated_mib"] == [1.0]
    assert view["vram_peak_reserved_mib"] == [4.0]
    assert view["estimated_gflops"] == [12.5]
    assert view["estimated_compute_identity"]["method"] == "test-estimator"


class AlternatingBenchmark(TrackingBenchmark):
    def _reset_episode(self, context):
        result = super()._reset_episode(context)
        self._terminal_outcome = "success" if context.rollout_index % 2 == 0 else "failure"
        return result


class InterruptedPolicy(TrackingPolicy):
    def _predict(self, observation):
        raise KeyboardInterrupt()


def _run_id(name):
    return f"{name}_2026-07-30_12-00-00_abcdef12"


def _snapshot():
    return RunConfigurationSnapshot(
        "schema_version: fixture\n",
        f"schema_version: fixture-resolved\nscientific_config_hash: {'a' * 64}\nexecution_config_hash: {'b' * 64}\n",
        "a" * 64, "b" * 64,
    )


def _canonical_run(
    root: Path, name="report-fixture", *, rollouts=1, alternating=False, requested=False,
    maximum_steps=3,
):
    run_id = _run_id(name)
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=5),
        rollout_count_per_task=rollouts,
        enabled_metric_ids=(
            "task.success", "task.success_rate", "action.variance",
            "action.smoothness_1", "action.smoothness_2",
            "system.inference_latency", "failure.collision_rate",
        ),
        metric_configurations={
            metric_id: ActionSequenceMetricConfig(
                ActionSource.REQUESTED if requested else ActionSource.APPLIED
            )
            for metric_id in ("action.variance", "action.smoothness_1", "action.smoothness_2")
        },
        metadata={"experiment_id": name, "experiment_name": name, "experiment_tags": ("report",)},
    )
    benchmark_type = AlternatingBenchmark if alternating else TrackingBenchmark
    runner = ExperimentRunner(
        plan, benchmark_type(maximum_steps=maximum_steps), TrackingPolicy(), FilesystemRunArtifactStore(root),
        clock=DeterministicClock(), configuration_snapshot=_snapshot(),
    )
    runner.connect(); runner.run()
    return run_id, root / run_id


def _report_profile_document():
    return builtin_profile().document()


def test_profile_schema_registry_and_template_path_safety(tmp_path):
    profile = ReportProfile.from_mapping(_report_profile_document())
    assert profile.identifier == "libero-task-default"
    invalid = _report_profile_document(); invalid["template"] = "../../secrets"
    with pytest.raises(ArtifactError, match="escapes"):
        ReportProfile.from_mapping(invalid, template_base=tmp_path)
    invalid = _report_profile_document(); invalid["charts"][0]["builder"] = "python_eval"
    with pytest.raises(ArtifactError, match="unknown report chart"):
        ReportProfile.from_mapping(invalid)
    engine = DerivedReportEngine(tmp_path / "runs", tmp_path / "derived")
    with pytest.raises(ArtifactError, match="run must be"):
        engine.resolve_run("../outside")


def test_local_template_bundle_is_configurable_but_path_independent(tmp_path):
    runs = tmp_path / "runs"
    run_id, _ = _canonical_run(runs, name="local-template")
    bundle = tmp_path / "templates" / "minimal"
    bundle.mkdir(parents=True)
    (bundle / "run-v1.html").write_text(
        '<!doctype html><html><body>LOCAL RUN {{ model.run.run_id }}</body></html>', encoding="utf-8"
    )
    (bundle / "task-v1.html").write_text(
        '<!doctype html><html><body>LOCAL TASK {{ task.task_id }}</body></html>', encoding="utf-8"
    )
    (bundle / "style.css").write_text("body { color: black; }\n", encoding="utf-8")
    document = _report_profile_document()
    document["id"] = "local-minimal"
    document["template"] = "minimal"
    profile = ReportProfile.from_mapping(document, template_base=bundle.parent)
    generated = DerivedReportEngine(runs, tmp_path / "derived", profile).generate(run_id)
    assert "LOCAL RUN" in (Path(generated["output"]) / "index.html").read_text(encoding="utf-8")
    assert profile.document()["template"] == "local-template-bundle"


def test_single_episode_report_is_binary_offline_traceable_and_tamper_evident(tmp_path):
    runs, derived = tmp_path / "runs", tmp_path / "derived"
    run_id, run_path = _canonical_run(runs)
    before = (run_path / "integrity.json").read_bytes()
    engine = DerivedReportEngine(runs, derived)
    generated = engine.generate(run_id)
    target = Path(generated["output"])
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    task = report["tasks"][0]
    assert task["outcome"] == {
        "eligible_episode_count": 1, "successful_episode_count": 1,
        "failed_episode_count": 0, "interrupted_or_invalid_episode_count": 0,
        "non_success_episode_count": 0, "terminal_status_counts": {"success": 1},
        "missing_episode_count": 0,
        "denominator_semantics": "all finalized episodes; missing episodes are displayed separately and never silently excluded",
        "presentation": "binary_success", "success": True, "success_rate": None,
    }
    assert report["schema_version"] == "ovlab.report/v1"
    chart_sources = {chart["builder"]: chart["canonical_sources"] for chart in report["charts"]}
    assert chart_sources["action_timeseries"] == ["tasks/*/episodes/*/trace.json:executed_actions.applied_action"]
    assert chart_sources["latency_distribution"] == ["tasks/*/episodes/*/trace.json:policy_predictions.inference_duration_ns"]
    chart_versions = {chart["builder"]: chart["builder_version"] for chart in report["charts"]}
    assert chart_versions == {
        "action_timeseries": "1.2.0", "latency_distribution": "1.2.0",
        "episode_outcomes": "1.2.0", "vram_timeseries": "1.1.0",
        "estimated_compute_timeseries": "1.1.0",
    }
    assert all(chart["interaction"]["runtime"] == "self-contained SVG; no network dependency" for chart in report["charts"])
    charts = {chart["builder"]: chart for chart in report["charts"]}
    action_statistics = charts["action_timeseries"]["statistics"]
    assert action_statistics["schema_version"] == "ovlab.descriptive-statistics/v1"
    assert "full canonical trace" in action_statistics["source"]
    assert {row["series"] for row in action_statistics["rows"]} == {"tx", "ty", "tz"}
    assert all(row["n"] == task["episodes"][0]["trace_view"]["action_sample_count"] for row in action_statistics["rows"])
    assert all(row["non_finite_count"] == 0 for row in action_statistics["rows"])
    assert all(row["sample_standard_deviation"] is not None for row in action_statistics["rows"])
    outcome_statistics = charts["episode_outcomes"]["statistics"]
    assert outcome_statistics["rows"][0]["n"] == 1
    assert outcome_statistics["rows"][0]["mean"] == 1.0
    assert outcome_statistics["rows"][0]["sample_standard_deviation"] is None
    assert all(metric["canonical_metric_ref"]["metric_id"] == metric["metric_id"] for metric in task["metrics"])
    unavailable = [metric for metric in task["metrics"] if metric["status"] != "available"]
    assert unavailable and all(metric["value"] is None for metric in unavailable)
    html = (target / "index.html").read_text(encoding="utf-8")
    task_html = next(target.glob("tasks/*/index.html")).read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert '<object class="chart" type="image/svg+xml"' in html
    assert "mouse wheel to zoom" in html
    assert "Tables use full canonical samples" in html
    assert "Sample SD" in html and "P05" in html and "P95" in html
    assert "binary episode outcome, not a statistical success rate" in task_html
    assert "Descriptive statistics" in task_html
    assert "success rate" not in task_html.lower().replace("not a statistical success rate", "")
    assert (run_path / "integrity.json").read_bytes() == before
    repeated = engine.generate(run_id)
    assert repeated["reused"] is True
    assert repeated["derived_build_id"] == generated["derived_build_id"]
    svg = target / "assets/charts/action_components.svg"
    svg_text = svg.read_text(encoding="utf-8")
    assert "Ordered control sample" in svg_text
    assert "Applied action (normalized command)" in svg_text
    assert 'id="interaction"' in svg_text
    assert "wheel: zoom" in svg_text and "drag: pan" in svg_text and "double-click: reset" in svg_text
    assert "<![CDATA[" in svg_text
    latency_svg = (target / "assets/charts/inference_latency.svg").read_text(encoding="utf-8")
    assert "Ordered prediction sample" in latency_svg and "Policy inference latency (ms)" in latency_svg
    outcomes_svg = (target / "assets/charts/episode_outcomes.svg").read_text(encoding="utf-8")
    assert "Episode index" in outcomes_svg and "Outcome (0=failure, 1=success)" in outcomes_svg
    svg.write_text(svg_text + "tamper", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        engine.verify(run_id, build_id=generated["derived_build_id"])


def test_report_manifest_tampering_is_detected(tmp_path):
    runs, derived = tmp_path / "runs", tmp_path / "derived"
    run_id, _ = _canonical_run(runs, name="manifest-tamper")
    engine = DerivedReportEngine(runs, derived)
    generated = engine.generate(run_id)
    manifest_path = Path(generated["output"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at_utc"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="manifest payload checksum"):
        engine.verify(run_id, build_id=generated["derived_build_id"])


def test_report_verification_rejects_tampered_canonical_source(tmp_path):
    runs, derived = tmp_path / "runs", tmp_path / "derived"
    run_id, run_path = _canonical_run(runs, name="canonical-tamper")
    engine = DerivedReportEngine(runs, derived)
    generated = engine.generate(run_id)
    plan = run_path / "plan.json"
    plan.write_bytes(plan.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="checksum|integrity"):
        engine.verify(run_id, build_id=generated["derived_build_id"])


def test_multi_episode_denominator_mixed_success_and_n_less_than_two_stddev(tmp_path):
    run_id, run_path = _canonical_run(tmp_path / "runs", name="mixed", rollouts=2, alternating=True)
    model = build_report_model(run_path)
    outcome = model["tasks"][0]["outcome"]
    assert outcome["presentation"] == "success_rate"
    assert outcome["eligible_episode_count"] == 2
    assert outcome["successful_episode_count"] == 1
    assert outcome["non_success_episode_count"] == 1
    assert outcome["success_rate"] == 0.5
    single_id, single_path = _canonical_run(tmp_path / "runs", name="single")
    single = build_report_model(single_path)
    summaries = [metric["value"] for metric in single["tasks"][0]["metrics"] if isinstance(metric["value"], dict)]
    assert summaries and all(value["standard_deviation"] is None for value in summaries)
    assert all(value["standard_deviation_qualification"] == "unavailable for n < 2" for value in summaries)


def test_time_limit_is_visible_non_success_with_terminal_reason_table(tmp_path):
    runs = tmp_path / "runs"
    run_id = _run_id("time-limit")
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=13),
        enabled_metric_ids=("task.success", "action.variance", "system.inference_latency"),
        metadata={"experiment_id": "time-limit"},
    )
    runner = ExperimentRunner(
        plan, TrackingBenchmark(maximum_steps=2, terminal_outcomes=("time_limit",)),
        TrackingPolicy(), FilesystemRunArtifactStore(runs), clock=DeterministicClock(),
        configuration_snapshot=_snapshot(),
    )
    runner.connect(); runner.run()

    model = build_report_model(runs / run_id)
    outcome = model["outcome"]
    assert outcome["successful_episode_count"] == 0
    assert outcome["non_success_episode_count"] == 1
    assert outcome["failed_episode_count"] == 0
    assert outcome["terminal_status_counts"] == {"time_limit": 1}
    episode = model["tasks"][0]["episodes"][0]
    assert episode["terminal_status"] == "time_limit"
    assert episode["terminal_reason"] == (
        "maximum episode step limit reached before benchmark success (2/2 executed steps)"
    )

    generated = DerivedReportEngine(runs, tmp_path / "derived").generate(run_id)
    root = Path(generated["output"])
    run_html = (root / "index.html").read_text(encoding="utf-8")
    task_html = next(root.glob("tasks/*/index.html")).read_text(encoding="utf-8")
    assert "Non-success: <strong>1</strong>" in run_html
    assert "<code>time_limit</code>=1" in run_html
    assert "Episode terminal outcomes" in run_html
    assert "maximum episode step limit reached before benchmark success" in run_html
    action_section = task_html.split("<h2>Action metrics</h2>", 1)[1].split("</section>", 1)[0]
    assert "valid_episode_count" not in action_section
    assert "<th>Mean</th>" in task_html and "<th>Sample SD</th>" in task_html


def test_failed_task_is_distinct_from_report_or_infrastructure_failure(tmp_path):
    runs = tmp_path / "runs"
    run_id = _run_id("task-failure")
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=9),
        enabled_metric_ids=("task.success",), metadata={"experiment_id": "task-failure"},
    )
    runner = ExperimentRunner(
        plan, TrackingBenchmark(maximum_steps=2, terminal_outcomes=("failure",)),
        TrackingPolicy(), FilesystemRunArtifactStore(runs), clock=DeterministicClock(),
        configuration_snapshot=_snapshot(),
    )
    runner.connect(); runner.run()
    model = build_report_model(runs / run_id)
    assert model["run"]["status"] == "completed"
    assert model["tasks"][0]["status"] == "failed"
    assert model["tasks"][0]["outcome"]["success"] is False
    generated = DerivedReportEngine(runs, tmp_path / "derived").generate(run_id)
    html = next(Path(generated["output"]).glob("tasks/*/index.html")).read_text(encoding="utf-8")
    assert "Success: <strong>false</strong>" in html
    assert "reporting failure" not in html.lower()


def test_interrupted_partial_run_renders_available_evidence(tmp_path):
    runs = tmp_path / "runs"
    run_id = _run_id("interrupted")
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=11),
        enabled_metric_ids=("task.success",), metadata={"experiment_id": "interrupted"},
    )
    runner = ExperimentRunner(
        plan, TrackingBenchmark(maximum_steps=3), InterruptedPolicy(),
        FilesystemRunArtifactStore(runs), clock=DeterministicClock(),
        configuration_snapshot=_snapshot(),
    )
    runner.connect()
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    model = build_report_model(runs / run_id)
    assert model["run"]["status"] == "aborted"
    assert model["tasks"][0]["status"] == "interrupted"
    assert model["tasks"][0]["partial"] is True
    generated = DerivedReportEngine(runs, tmp_path / "derived").generate(run_id)
    task_html = next(Path(generated["output"]).glob("tasks/*/index.html")).read_text(encoding="utf-8")
    assert "partial or interrupted" in task_html


def test_automatic_partial_then_final_report_lifecycle(tmp_path):
    runs, derived = tmp_path / "runs", tmp_path / "derived"
    run_id = _run_id("automatic")
    plan = runner_plan(
        run_context=make_run_context(run_id=run_id, seed=7),
        enabled_metric_ids=("task.success",), metadata={"experiment_id": "automatic"},
    )
    reporter = AutomaticDerivedReporter(DerivedReportEngine(runs, derived))
    runner = ExperimentRunner(
        plan, TrackingBenchmark(maximum_steps=2), TrackingPolicy(), FilesystemRunArtifactStore(runs),
        clock=DeterministicClock(), postprocessor=reporter, configuration_snapshot=_snapshot(),
    )
    runner.connect(); runner.run()
    profile_root = derived / run_id / "libero-task-default"
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in profile_root.glob("*/manifest.json")]
    assert {item["status"] for item in manifests} == {"partial", "complete"}
    latest = json.loads((profile_root / "latest.json").read_text(encoding="utf-8"))
    assert latest["status"] == "complete"
    assert (runs / run_id / "manifest.completed.json").is_file()


def test_reporting_operational_failure_state_is_derived_only(tmp_path):
    runs, derived = tmp_path / "runs", tmp_path / "derived"
    run_id, run_path = _canonical_run(runs, name="report-failure-state")
    before = (run_path / "integrity.json").read_bytes()
    engine = DerivedReportEngine(runs, derived)
    engine.record_failure(run_id, "task", RuntimeError("template failure"))
    failure = next((derived / run_id / "libero-task-default/operational-failures").glob("*.json"))
    document = json.loads(failure.read_text(encoding="utf-8"))
    assert document["canonical_status_authoritative"] is True
    assert document["error_type"] == "RuntimeError"
    assert (run_path / "integrity.json").read_bytes() == before


def test_isolated_episode_and_run_exports_are_readable_atomic_units(tmp_path):
    runs, exports = tmp_path / "runs", tmp_path / "exports"
    run_id, _ = _canonical_run(runs, name="isolated", rollouts=2, alternating=True)
    model = build_report_model(runs / run_id)
    episode_id = model["tasks"][0]["episodes"][0]["episode_id"]
    engine = ExportEngine(runs, exports)

    episode_result = engine.generate_isolated(run_id, episode_id=episode_id)
    target = Path(episode_result["output"])
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert target == exports / "isolated" / run_id
    assert metadata["export_type"] == "isolated"
    assert metadata["source"]["scope"] == "episode"
    assert metadata["source"]["episodes"] == [episode_id]
    assert set(metadata) >= {"source", "experiment", "model", "checkpoint", "config", "template", "datetime"}
    assert len(list((target / "episodes/tables").glob("*__statistics.csv"))) == 1
    assert len(list((target / "episodes/tables").glob("*__timeseries.csv"))) == 1
    assert len(list((target / "episodes/tables").glob("*__action-metrics.csv"))) == 1
    assert list((target / "episodes/figures").glob("*__actions-over-time.png"))
    assert list((target / "episodes/figures").glob("*__actions-over-time.pdf"))
    assert list((target / "episodes/figures").glob("*__action-metrics.png"))
    assert list((target / "episodes/figures").glob("*__action-metrics.pdf"))
    assert not (target / "overview").exists()
    assert not (target / "manifest.json").exists() and not (target / "export.json").exists()

    run_result = engine.generate_isolated(run_id)
    target = Path(run_result["output"])
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"]["scope"] == "run"
    assert len(metadata["source"]["episodes"]) == 2
    assert (target / "overview/tables/episode-summary.csv").is_file()
    assert (target / "overview/tables/descriptive-statistics.csv").is_file()
    assert (target / "overview/tables/metric-summary.csv").is_file()
    assert (target / "overview/tables/action-metrics-by-episode.csv").is_file()
    assert (target / "overview/tables/action-metrics-summary.csv").is_file()
    assert (target / "overview/figures/success-by-task.png").is_file()
    assert (target / "overview/figures/action-boxplots.pdf").is_file()
    assert (target / "overview/figures/action-metrics-by-episode.png").is_file()
    assert (target / "overview/figures/action-metrics-by-task.pdf").is_file()
    with (target / "overview/tables/action-metrics-by-episode.csv").open(newline="") as stream:
        action_rows = list(csv.DictReader(stream))
    assert {row["metric_id"] for row in action_rows} == {
        "action.variance", "action.smoothness_1", "action.smoothness_2",
    }
    assert all(row["status"] == "available" for row in action_rows)
    with (target / "overview/tables/action-metrics-summary.csv").open(newline="") as stream:
        action_summary = list(csv.DictReader(stream))
    assert {row["scope"] for row in action_summary} == {"run", "task"}
    assert all(row["available_count"] == "2" for row in action_summary)
    statistics_text = (target / "overview/tables/descriptive-statistics.csv").read_text(encoding="utf-8")
    assert "scope" in statistics_text.splitlines()[0]
    assert any(line.startswith("run,") for line in statistics_text.splitlines()[1:])
    assert engine.verify("isolated", run_id)["integrity"] == "verified"

    table = target / "overview/tables/episode-summary.csv"
    table.write_bytes(table.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        engine.verify("isolated", run_id)


def test_grouped_manual_all_and_same_model_exports(tmp_path):
    runs, exports = tmp_path / "runs", tmp_path / "exports"
    first, _ = _canonical_run(runs, name="group-one", rollouts=2, alternating=True)
    second, _ = _canonical_run(runs, name="group-two", rollouts=2, alternating=True)
    engine = ExportEngine(runs, exports)
    generated = engine.generate_grouped("manual-comparison", run_ids=[second, first])
    target = Path(generated["output"])
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert target == exports / "grouped/manual-comparison"
    assert metadata["source"]["runs"] == sorted([first, second])
    assert metadata["source"]["selection"]["mode"] == "manual"
    assert (target / "tables/run-summary.csv").is_file()
    assert (target / "tables/episode-summary.csv").is_file()
    assert (target / "tables/descriptive-statistics.csv").is_file()
    assert (target / "tables/metric-summary.csv").is_file()
    assert (target / "tables/action-metrics-by-episode.csv").is_file()
    assert (target / "tables/action-metrics-summary.csv").is_file()
    statistics_text = (target / "tables/descriptive-statistics.csv").read_text(encoding="utf-8")
    assert any(line.startswith("run,") for line in statistics_text.splitlines()[1:])
    assert any(line.startswith("group,") for line in statistics_text.splitlines()[1:])
    for name in (
        "success-comparison", "task-success-heatmap", "inference-latency-boxplots",
        "inference-latency-ecdf", "success-latency-pareto", "terminal-outcome-composition",
        "action-metrics-by-run", "action-metrics-by-model",
    ):
        assert (target / f"figures/{name}.png").is_file()
        assert (target / f"figures/{name}.pdf").is_file()
    assert engine.verify("grouped", "manual-comparison")["source_run_ids"] == sorted([first, second])
    with (target / "tables/action-metrics-summary.csv").open(newline="") as stream:
        action_summary = list(csv.DictReader(stream))
    assert {row["scope"] for row in action_summary} == {"run", "model", "group"}
    assert {row["metric_id"] for row in action_summary} == {
        "action.variance", "action.smoothness_1", "action.smoothness_2",
    }

    same = engine.generate_grouped("same-model", same_model_as=first)
    assert same["source_run_ids"] == sorted([first, second])
    all_runs = engine.generate_grouped("all-runs", all_runs=True)
    assert all_runs["source_run_ids"] == sorted([first, second])
    with pytest.raises(ArtifactError, match="exactly one selector"):
        engine.generate_grouped("invalid", all_runs=True, run_ids=[first])
    with pytest.raises(ArtifactError, match="unknown grouped export template"):
        engine.generate_grouped("invalid-template", run_ids=[first], template="made-up")


def test_action_metric_exports_preserve_unavailable_status_and_reason(tmp_path):
    runs, exports = tmp_path / "runs", tmp_path / "exports"
    run_id, _ = _canonical_run(runs, name="short-action", maximum_steps=1)
    generated = ExportEngine(runs, exports).generate_isolated(run_id)
    table = next((Path(generated["output"]) / "episodes/tables").glob("*__action-metrics.csv"))
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["metric_id"] for row in rows} == {
        "action.variance", "action.smoothness_1", "action.smoothness_2",
    }
    assert all(row["status"] == "insufficient_data" for row in rows)
    assert all(row["value"] == "" for row in rows)
    assert all(row["reason"] for row in rows)
    summary = Path(generated["output"]) / "overview/tables/action-metrics-summary.csv"
    with summary.open(newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert all(row["available_count"] == "0" for row in summary_rows)
    assert all(row["insufficient_data_count"] == "1" for row in summary_rows)
    assert all(row["unavailable_count"] == "0" for row in summary_rows)


def test_grouped_export_rejects_incompatible_metrics_empty_selection_and_unsafe_legacy_spec(tmp_path):
    runs = tmp_path / "runs"
    first, _ = _canonical_run(runs, name="compatible")
    second, _ = _canonical_run(runs, name="requested", requested=True)
    engine = ExportEngine(runs, tmp_path / "exports")
    with pytest.raises(ArtifactError, match="incompatible metric"):
        engine.generate_grouped("incompatible", run_ids=[first, second])
    with pytest.raises(ArtifactError, match="selection is empty"):
        engine.generate_grouped("empty", all_runs=True, suite="absent")
    invalid = {
        "schema_version": "ovlab.export-spec/v1", "id": "unsafe",
        "selection": {"run_ids": [first]}, "python": "eval('bad')",
    }
    with pytest.raises(ArtifactError, match="unsupported fields"):
        validate_export_spec(invalid)


def test_export_verification_rechecks_complete_canonical_source(tmp_path):
    runs = tmp_path / "runs"
    run_id, run_path = _canonical_run(runs, name="export-source-tamper")
    engine = ExportEngine(runs, tmp_path / "exports")
    engine.generate_isolated(run_id)
    trace = next(run_path.glob("tasks/*/episodes/*/trace.json"))
    trace.write_bytes(trace.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="canonical export source verification failed"):
        engine.verify("isolated", run_id)


def test_external_publish_can_skip_html_but_keeps_isolated_export_and_read_only_source(tmp_path):
    data = tmp_path / "data"
    run_id, run_path = _canonical_run(data / "runs", name="external-publish")
    before = (run_path / "integrity.json").read_bytes()
    app = OvlabApplication(REPOSITORY, environment={
        "OVLAB_DATA_ROOT": str(data),
        "OVLAB_RUNS_ROOT": str(data / "runs"),
        "OVLAB_DERIVED_ROOT": str(data / "derived"),
        "OVLAB_EXPORTS_ROOT": str(data / "exports"),
    })

    result = app.report_publish(run_id, report_enabled=False)

    assert result["report"] is None
    assert result["isolated_export"]["integrity"] == "verified"
    assert not (data / "derived").exists()
    assert (run_path / "integrity.json").read_bytes() == before


def test_unified_cli_report_and_export_json_workflows(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    data = tmp_path / "data"
    run_id, _ = _canonical_run(data / "runs", name="cli-report")
    environment = {
        **os.environ,
        "OVLAB_PYTHON": sys.executable,
        "OVLAB_DATA_ROOT": str(data),
        "OVLAB_RUNS_ROOT": str(data / "runs"),
        "OVLAB_DERIVED_ROOT": str(data / "derived"),
        "OVLAB_EXPORTS_ROOT": str(data / "exports"),
        "OVLAB_REPORTING_RUNTIME": "host",
    }

    def invoke(*arguments):
        return subprocess.run(
            [str(repository / "ovlab"), *arguments], cwd=repository, env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    profiles = invoke("report", "profiles", "--json")
    assert profiles.returncode == 0 and profiles.stderr == ""
    assert json.loads(profiles.stdout)["result"]["profiles"][0]["id"] == "libero-task-default"
    missing = invoke("report", "generate", "--run", "missing-run", "--json")
    assert missing.returncode == 4
    assert json.loads(missing.stdout)["errors"][0]["code"] == "source_unavailable"
    invalid_profile = tmp_path / "invalid-profile.yaml"
    invalid_profile.write_text("schema_version: wrong\n", encoding="utf-8")
    invalid = invoke("report", "generate", "--run", run_id, "--profile", str(invalid_profile), "--json")
    assert invalid.returncode == 3
    assert json.loads(invalid.stdout)["errors"][0]["code"] == "configuration_error"
    generated = invoke("report", "generate", "--run", run_id, "--profile", "libero-task-default", "--json")
    assert generated.returncode == 0 and generated.stderr == ""
    result = json.loads(generated.stdout)["result"]
    assert Path(result["output"]).is_relative_to(data / "derived")
    verified = invoke("report", "verify", "--run", run_id, "--profile", "libero-task-default", "--build", result["derived_build_id"], "--json")
    assert verified.returncode == 0 and json.loads(verified.stdout)["result"]["integrity"] == "verified"
    task = invoke("report", "generate", "--run", run_id, "--task", "mock-task-0", "--profile", "libero-task-default", "--json")
    assert task.returncode == 0 and json.loads(task.stdout)["result"]["integrity"] == "verified"

    published = invoke(
        "report", "publish", "--run", run_id, "--profile", "libero-task-default",
        "--report-enabled", "true", "--json",
    )
    assert published.returncode == 0 and published.stderr == ""
    publication = json.loads(published.stdout)["result"]
    assert publication["canonical_run_modified"] is False
    assert publication["report"]["integrity"] == "verified"
    assert publication["isolated_export"]["export_type"] == "isolated"

    isolated = invoke("export", "isolated", "--run", run_id, "--json")
    assert isolated.returncode == 0 and isolated.stderr == ""
    isolated_result = json.loads(isolated.stdout)["result"]
    assert Path(isolated_result["output"]) == data / "exports/isolated" / run_id
    checked = invoke("export", "verify", "--kind", "isolated", "--name", run_id, "--json")
    assert checked.returncode == 0 and json.loads(checked.stdout)["result"]["integrity"] == "verified"

    grouped = invoke("export", "grouped", "--name", "cli-group", "--runs", run_id, "--json")
    assert grouped.returncode == 0 and grouped.stderr == ""
    group_result = json.loads(grouped.stdout)["result"]
    assert Path(group_result["output"]) == data / "exports/grouped/cli-group"
    checked = invoke("export", "verify", "--kind", "grouped", "--name", "cli-group", "--json")
    assert checked.returncode == 0 and json.loads(checked.stdout)["result"]["integrity"] == "verified"
