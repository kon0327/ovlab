import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ovlab_benchctl.report_auto import ReportSelectionError, generate_report
from ovlab_benchctl.cli import _compact


ROOT = Path(__file__).resolve().parents[4]


def _runner(calls):
    def run(args, **kwargs):
        calls.append((args, kwargs["env"]))
        kind = args[2]
        category = args[args.index("--category") + 1]
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "status": "success", "result": {"output": f"/exports/{category}/{kind}",
                                             "integrity": "verified", "reused": False},
        }))
    return run


def _plan(manifest, experiment_id, nodes):
    (manifest.parent / "plan.json").write_text(json.dumps({
        "experiment_id": experiment_id, "id": "study", "nodes": nodes,
    }))


def _node(node_id, path, children=()):
    return {"id": node_id, "path": path, "children": list(children)}


def test_experiment_uses_grouped_export_for_each_multi_run_category(tmp_path):
    manifest = tmp_path / "runs/study-1/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "experiment_id": "study-1", "status": "completed", "nodes": {
            "a": {"train": {"run_id": "train-a"}, "evaluate": {"run_id": "eval-a"}},
            "b": {"train": {"run_id": "train-b"}, "evaluate": {"run_id": "eval-b"}},
        },
    }))
    _plan(manifest, "study-1", [_node("study", "", ("a", "b")), _node("a", "a"), _node("b", "b")])
    calls = []
    result = generate_report("study-1", repository_root=ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert result["source_type"] == "experiment" and result["source"] == "study-1"
    grouped = [item for item in result["exports"] if item["type"] == "grouped"]
    isolated = [item for item in result["exports"] if item["type"] == "isolated"]
    assert len(isolated) == 4
    assert [(item["category"], item["node"], item["runs"]) for item in grouped] == [
        ("train", "", ["train-a", "train-b"]),
        ("evaluate", "", ["eval-a", "eval-b"]),
    ]
    assert all(call[1]["OVLAB_EXPERIMENT_ID"] == "study-1" for call in calls)
    human = _compact("report auto", result)
    assert "study-1" in human and "train  grouped export" in human
    assert "eval-a, eval-b" in human and "/exports/evaluate/grouped" in human


def test_single_run_uses_isolated_export(tmp_path):
    run = tmp_path / "runs/evaluate/run-1"
    run.mkdir(parents=True)
    (run / "manifest.completed.json").write_text("{}")
    calls = []
    result = generate_report("run-1", repository_root=ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert result["source_type"] == "run"
    assert result["exports"][0]["type"] == "isolated"
    assert result["exports"][0]["runs"] == ["run-1"]
    assert calls[0][0][2] == "isolated"


def test_incomplete_experiment_is_not_silently_exported(tmp_path):
    manifest = tmp_path / "runs/study-1/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"experiment_id": "study-1", "status": "failed", "nodes": {}}))
    calls = []
    with pytest.raises(ReportSelectionError, match="not completed"):
        generate_report("study-1", repository_root=ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert not calls


def test_experiment_with_one_run_per_category_uses_separate_isolated_exports(tmp_path):
    manifest = tmp_path / "runs/study-2/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"experiment_id": "study-2", "status": "completed", "nodes": {
        "model": {"train": {"run_id": "train-1"}, "evaluate": {"run_id": "eval-1"}},
    }}))
    _plan(manifest, "study-2", [_node("study", "", ("model",)), _node("model", "model")])
    calls = []
    result = generate_report("study-2", repository_root=ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert [(item["category"], item["type"]) for item in result["exports"]] == [
        ("train", "isolated"), ("evaluate", "isolated")
    ]
    assert all(call[0][2] == "isolated" for call in calls)


def test_training_run_uses_training_export(tmp_path):
    run = tmp_path / "runs/train/train-1"
    run.mkdir(parents=True)
    (run / "training-plan.json").write_text("{}")
    calls = []
    result = generate_report("train-1", repository_root=ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert result["exports"][0]["category"] == "train"
    assert "train" in calls[0][0]


def test_nested_experiment_exports_each_run_and_each_parent_comparison(tmp_path):
    manifest = tmp_path / "runs/study-nested/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"experiment_id": "study-nested", "status": "completed", "nodes": {
        "left/a": {"evaluate": {"run_id": "run-a"}},
        "left/b": {"evaluate": {"run_id": "run-b"}},
        "right/c": {"evaluate": {"run_id": "run-c"}},
        "right/d": {"evaluate": {"run_id": "run-d"}},
    }}))
    _plan(manifest, "study-nested", [
        _node("study", "", ("left", "right")),
        _node("left", "left", ("a", "b")), _node("a", "left/a"), _node("b", "left/b"),
        _node("right", "right", ("c", "d")), _node("c", "right/c"), _node("d", "right/d"),
    ])
    calls = []
    result = generate_report("study-nested", repository_root=ROOT,
                             environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    individual = {item["node"]: item["runs"] for item in result["exports"] if item["type"] == "isolated"}
    grouped = {item["node"]: item["runs"] for item in result["exports"] if item["type"] == "grouped"}
    assert individual == {"left/a": ["run-a"], "left/b": ["run-b"], "right/c": ["run-c"], "right/d": ["run-d"]}
    assert grouped == {"left": ["run-a", "run-b"], "right": ["run-c", "run-d"],
                       "": ["run-a", "run-b", "run-c", "run-d"]}
    group_calls = {env["OVLAB_EXPERIMENT_NODE"]: args for args, env in calls if args[2] == "grouped"}
    assert group_calls["left"][group_calls["left"].index("--name") + 1] == "left-comparison"
    assert group_calls[""][group_calls[""].index("--name") + 1] == "study-comparison"
    assert "node       left" in _compact("report auto", result)


def test_parent_own_run_is_isolated_and_not_mixed_into_child_comparison(tmp_path):
    manifest = tmp_path / "runs/study-direct/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"experiment_id": "study-direct", "status": "completed", "nodes": {
        "": {"evaluate": {"run_id": "parent-run"}},
        "a": {"evaluate": {"run_id": "child-a"}},
        "b": {"evaluate": {"run_id": "child-b"}},
    }}))
    _plan(manifest, "study-direct", [_node("study", "", ("a", "b")), _node("a", "a"), _node("b", "b")])
    result = generate_report("study-direct", repository_root=ROOT,
                             environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner([]))
    grouped = [item for item in result["exports"] if item["type"] == "grouped"]
    assert len(grouped) == 1 and grouped[0]["runs"] == ["child-a", "child-b"]
    assert any(item["type"] == "isolated" and item["runs"] == ["parent-run"] for item in result["exports"])


def test_completed_experiment_without_plan_does_not_guess_groups(tmp_path):
    manifest = tmp_path / "runs/study-missing/.experiment/result.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"experiment_id": "study-missing", "status": "completed", "nodes": {
        "a": {"evaluate": {"run_id": "run-a"}},
    }}))
    calls = []
    with pytest.raises(ReportSelectionError, match="plan is unavailable"):
        generate_report("study-missing", repository_root=ROOT,
                        environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=_runner(calls))
    assert not calls
