"""Dependency-light contracts for the hierarchical experiment owner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ovlab_benchctl.experiment_workflow import ExperimentExecutionError, ExperimentSpecError, ExperimentWorkflow, build_experiment_index
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.training_composition import load_training_profile
from ovlab_benchctl.strict_yaml import dumps


ROOT = Path(__file__).resolve().parents[4]


def _spec():
    return {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow",
        "id": "models-on-libero", "name": "Models on shared LIBERO tasks",
        "config": {
            "train": {"profile": "openvla-lora-libero10-smoke.yaml", "patch": {"training": {"seed": 0}}},
            "evaluate": {"profile": "libero10-lora-4bit-three-per-task.yaml"},
        },
        "children": [
            {"id": "model-a", "execute": ["evaluate"], "children": [
                {"id": "task-a", "config": {"evaluate": {"component_patches": {
                    "benchmark": {"settings": {"task_indices": [0]}}
                }}}, "execute": ["train", "evaluate"]},
            ]},
            {"id": "model-b", "execute": ["evaluate"], "config": {"train": {
                "patch": {"training": {"learning_rate": 0.00025}}
            }}},
        ],
    }


def _write(tmp_path, document):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_checked_in_yaml_and_json_examples_describe_the_same_experiment():
    workflow = ExperimentWorkflow(ROOT)
    directory = ROOT / "experiments/workflows"
    json_plan = workflow.plan(directory / "libero-model-tree.json")
    yaml_plan = workflow.plan(directory / "libero-model-tree.yaml")
    json_plan.pop("spec")
    yaml_plan.pop("spec")
    assert yaml_plan == json_plan


def test_reference_examples_validate_and_resolve_together():
    plan = ExperimentWorkflow(ROOT).plan("experiments/workflows/example.yaml")
    assert [step["phase"] for step in plan["steps"]] == ["evaluate", "evaluate"]
    configs = plan["nodes"][0]["configuration"]
    assert configs["train"]["profile"] == "experiments/training/example.yaml"
    assert configs["evaluate"]["profile"] == "experiments/evaluation/example.yaml"
    train = load_training_profile(ROOT / "experiments/training/example.yaml")
    assert train.document["training"]["mode"] == "peft"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("kind", [None, "experiment", "", 1])
def test_workflow_kind_is_required_and_must_match(tmp_path, suffix, kind):
    document = _spec()
    if kind is None:
        del document["kind"]
    else:
        document["kind"] = kind
    path = tmp_path / ("workflow" + suffix)
    path.write_text(json.dumps(document) if suffix == ".json" else dumps(document), encoding="utf-8")
    with pytest.raises(ExperimentSpecError, match="kind"):
        ExperimentWorkflow(ROOT).plan(path)


def test_workflow_kind_is_not_allowed_on_child_nodes(tmp_path):
    document = _spec()
    document["children"][0]["kind"] = "workflow"
    with pytest.raises(ExperimentSpecError, match="kind"):
        ExperimentWorkflow(ROOT).plan(_write(tmp_path, document))


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_yaml_and_json_have_identical_validation_inheritance_and_plan(tmp_path, suffix):
    document = _spec()
    json_path = _write(tmp_path, document)
    yaml_path = tmp_path / ("experiment" + suffix)
    yaml_path.write_text("# Human-readable workflow\n" + dumps(document), encoding="utf-8")
    workflow = ExperimentWorkflow(ROOT)
    json_plan = workflow.plan(json_path)
    yaml_plan = workflow.plan(yaml_path)
    assert yaml_plan.pop("spec") == str(yaml_path)
    assert json_plan.pop("spec") == str(json_path)
    assert yaml_plan == json_plan
    assert yaml_plan["nodes"][2]["configuration"]["train"]["document"]["training"]["seed"] == 0


@pytest.mark.parametrize("source", [
    "schema_version: ovlab.experiment/v1\nkind: workflow\nid: duplicate\nid: other\nname: Example\n",
    "schema_version: ovlab.experiment/v1\nkind: workflow\nid: example\nname: &alias Example\n",
    "schema_version: ovlab.experiment/v1\nkind: workflow\nid: example\nname: Example\nunknown: true\n",
])
def test_invalid_yaml_fails_before_creating_artifacts(tmp_path, source):
    path = tmp_path / "invalid.yaml"
    path.write_text(source, encoding="utf-8")
    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path / "data")})
    with pytest.raises(ExperimentSpecError):
        workflow.run(path)
    assert not (tmp_path / "data").exists()


def test_yaml_run_preserves_original_source_and_canonical_json(tmp_path):
    document = {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "yaml-test", "name": "YAML workflow",
        "config": {"evaluate": {"profile": "mock-e2e-smoke.yaml"}}, "execute": ["evaluate"],
    }
    path = tmp_path / "experiment.yml"
    original = "# Preserve this comment\n" + dumps(document)
    path.write_text(original, encoding="utf-8")

    def fake_runner(command, **kwargs):
        instance = kwargs["env"]["OVLAB_EXPERIMENT_ID"]
        result = ({"run_path": str(tmp_path / "runs" / instance / "evaluate" / "run-one")}
                  if command[1] == "run" else {"status": "completed", "report": {"interactive_output": "report.html"}})
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=fake_runner)
    result = workflow.run(path)
    metadata = tmp_path / "runs" / result["experiment_id"] / ".experiment"
    assert (metadata / "spec.yaml").read_text(encoding="utf-8") == original
    assert json.loads((metadata / "spec.json").read_text(encoding="utf-8")) == document
    assert json.loads((metadata / "plan.json").read_text(encoding="utf-8"))["spec_hash"] == result["spec_hash"]
    assert path.read_text(encoding="utf-8") == original


def test_yaml_changed_after_planning_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "experiment.yaml"
    path.write_text(dumps(_spec()), encoding="utf-8")
    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path / "data")})
    plan = workflow.plan(path)
    altered = _spec()
    altered["name"] = "Changed"
    path.write_text(dumps(altered), encoding="utf-8")
    monkeypatch.setattr(workflow, "plan", lambda *args, **kwargs: plan)
    with pytest.raises(ExperimentSpecError, match="changed during planning"):
        workflow.run(path)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("suffix,content", [
    (".json", '{"schema_version":"ovlab.experiment/v1","kind":"workflow","id":"one","id":"two","name":"Test"}'),
    (".yaml", "schema_version: ovlab.experiment/v1\nkind: workflow\nid: one\nid: two\nname: Test\n"),
])
def test_invalid_specs_return_the_same_cli_configuration_error(tmp_path, suffix, content):
    path = tmp_path / ("invalid" + suffix)
    path.write_text(content, encoding="utf-8")
    completed = subprocess.run(
        [str(ROOT / "ovlab"), "experiment", "plan", str(path), "--json"],
        cwd=ROOT, env={**os.environ, "OVLAB_PYTHON": sys.executable},
        text=True, capture_output=True,
    )
    assert completed.returncode == 3
    assert json.loads(completed.stdout)["errors"][0]["code"] == "configuration_error"


def test_plan_inherits_resolved_profiles_and_tracks_field_provenance(tmp_path):
    plan = ExperimentWorkflow(ROOT).plan(_write(tmp_path, _spec()))
    assert plan["schema_version"] == "ovlab.experiment-plan/v1"
    assert [step["node"] + ":" + step["phase"] for step in plan["steps"]] == [
        "model-a:evaluate", "model-a/task-a:train", "model-a/task-a:evaluate", "model-b:evaluate",
    ]
    nodes = {node["path"]: node for node in plan["nodes"]}
    trained = nodes["model-a/task-a"]["configuration"]["train"]
    assert trained["document"]["training"]["seed"] == 0
    assert trained["field_sources"]["training.seed"] == "experiment:<root>:training.seed"
    assert nodes["model-b"]["configuration"]["train"]["document"]["training"]["learning_rate"] == 0.00025
    benchmark = nodes["model-a/task-a"]["configuration"]["evaluate"]["component_documents"]["benchmark"]
    assert benchmark["document"]["settings"]["task_indices"] == [0]
    assert benchmark["field_sources"]["settings.task_indices"] == "experiment:model-a/task-a:settings.task_indices"
    assert plan["dependency_semantics"].startswith("scheduling only")


def test_plan_can_select_one_node_without_descendants_or_external_dependency(tmp_path):
    spec = _spec()
    path = _write(tmp_path, spec)
    plan = ExperimentWorkflow(ROOT).plan(path, node="model-a", node_only=True)
    assert plan["steps"] == [{"node": "model-a", "phase": "evaluate"}]
    spec["children"][0]["depends_on"] = [{"node": "model-b", "phase": "evaluate"}]
    with pytest.raises(ExperimentSpecError, match="unselected"):
        ExperimentWorkflow(ROOT).plan(_write(tmp_path, spec), node="model-a")


def test_standalone_node_is_not_blocked_by_an_unrelated_sibling_profile(tmp_path):
    spec = _spec()
    spec["children"][1]["config"] = {"evaluate": {"profile": "missing-sibling.yaml"}}
    path = _write(tmp_path, spec)
    selected = ExperimentWorkflow(ROOT).plan(path, node="model-a")
    assert [step["node"] for step in selected["steps"]] == ["model-a", "model-a/task-a", "model-a/task-a"]
    with pytest.raises(ExperimentSpecError, match="profile must be a regular file"):
        ExperimentWorkflow(ROOT).plan(path)


@pytest.mark.parametrize("mutate,reason", [
    (lambda spec: spec["children"].append({"id": "model-a"}), "duplicate child"),
    (lambda spec: spec["children"][0].update(id="../outside"), "portable identifier"),
    (lambda spec: spec["children"][0].update(id="train"), "portable identifier"),
    (lambda spec: spec["children"][0].update(config={"evaluate": {"patch": {"unknown": True}}}), "unknown"),
])
def test_invalid_tree_or_patch_is_rejected_before_execution(tmp_path, mutate, reason):
    spec = _spec()
    mutate(spec)
    with pytest.raises(Exception, match=reason):
        ExperimentWorkflow(ROOT).plan(_write(tmp_path, spec))


def test_dependency_cycle_is_rejected(tmp_path):
    spec = _spec()
    spec["children"][0]["depends_on"] = [{"node": "model-b", "phase": "evaluate"}]
    spec["children"][1]["depends_on"] = [{"node": "model-a", "phase": "evaluate"}]
    with pytest.raises(ExperimentSpecError, match="cycle"):
        ExperimentWorkflow(ROOT).plan(_write(tmp_path, spec))


def test_run_scopes_children_publishes_grouped_and_cleans_ephemeral_yaml(tmp_path):
    spec = _spec()
    spec.pop("config")
    spec["config"] = {"evaluate": {"profile": "libero10-lora-4bit-three-per-task.yaml"}}
    spec["children"][0].pop("children")
    spec["children"][1].pop("config")
    calls = []

    def fake_runner(command, **kwargs):
        env = kwargs["env"]
        calls.append((command, env.copy()))
        node = env["OVLAB_EXPERIMENT_NODE"]
        instance = env["OVLAB_EXPERIMENT_ID"]
        if command[1:3] == ["export", "grouped"]:
            result = {"output": str(tmp_path / "exports" / instance / "evaluate" / "grouped" / "index.html")}
        else:
            result = {"run_path": str(tmp_path / "runs" / instance / "evaluate" / node / (node + "-run"))}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    env = {"OVLAB_RUNS_ROOT": str(tmp_path / "runs"), "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports")}
    workflow = ExperimentWorkflow(ROOT, environment=env, runner=fake_runner)
    result = workflow.run(_write(tmp_path, spec), sequential=True)
    assert result["status"] == "completed"
    assert [call[0][1:3] for call in calls] == [["eval", "run"], ["eval", "run"], ["export", "grouped"]]
    assert [call[1]["OVLAB_EXPERIMENT_NODE"] for call in calls] == ["model-a", "model-b", ""]
    assert result["grouped"]["evaluate"][""]["status"] == "completed"
    manifest = tmp_path / "runs" / result["experiment_id"] / ".experiment"
    assert json.loads((manifest / "plan.json").read_text())["nodes"][1]["configuration"]["evaluate"]["document"]["kind"] == "experiment"
    assert json.loads((manifest / "result.json").read_text())["status"] == "completed"
    assert (tmp_path / "exports" / result["experiment_id"] / "experiment.json").is_file()
    overview = (tmp_path / "exports" / result["experiment_id"] / "index.html").read_text()
    assert "Models on shared LIBERO tasks" in overview
    assert "model-a" in overview and "model-b" in overview
    assert "<details" not in overview
    assert (tmp_path / "runs").stat().st_mode & 0o7777 == 0o2770
    assert (tmp_path / "exports").stat().st_mode & 0o7777 == 0o2770
    assert not (ROOT / "experiments/evaluation/.generated" / result["experiment_id"]).exists()


def test_independent_phases_run_in_parallel_and_dependencies_wait(tmp_path):
    spec = {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "parallel", "name": "Parallel",
        "config": {"evaluate": {"profile": "libero10-lora-4bit-three-per-task.yaml"}},
        "children": [
            {"id": "a", "execute": ["evaluate"]},
            {"id": "b", "execute": ["evaluate"]},
            {"id": "c", "execute": ["evaluate"], "depends_on": [{"node": "a", "phase": "evaluate"}]},
        ],
    }
    barrier = threading.Barrier(2, timeout=5)
    finished_a = threading.Event()
    seen = []

    def fake_runner(command, **kwargs):
        env = kwargs["env"]
        node = env["OVLAB_EXPERIMENT_NODE"]
        if command[1:3] == ["export", "grouped"]:
            result = {"output": str(tmp_path / "exports" / env["OVLAB_EXPERIMENT_ID"] / "grouped")}
        else:
            seen.append(node)
            if node in {"a", "b"}:
                barrier.wait()
                if node == "a":
                    finished_a.set()
            else:
                assert finished_a.is_set()
            result = {"run_path": str(tmp_path / "runs" / env["OVLAB_EXPERIMENT_ID"] / "evaluate" / node / f"{node}-run")}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=fake_runner)
    result = workflow.run(_write(tmp_path, spec))
    assert set(seen) == {"a", "b", "c"}
    assert seen.index("c") > seen.index("a")
    assert result["execution"] == {"mode": "parallel", "max_workers": 2}
    assert set(result["nodes"]) == {"a", "b", "c"}
    assert result["grouped"]["evaluate"][""]["status"] == "completed"


def test_sequential_mode_runs_one_phase_at_a_time(tmp_path):
    spec = _spec()
    spec["children"][0].pop("children")
    running = 0
    maximum = 0
    lock = threading.Lock()

    def fake_runner(command, **kwargs):
        nonlocal running, maximum
        env = kwargs["env"]
        if command[1:3] == ["export", "grouped"]:
            result = {"output": str(tmp_path / "grouped")}
        else:
            with lock:
                running += 1
                maximum = max(maximum, running)
            time.sleep(0.02)
            with lock:
                running -= 1
            node = env["OVLAB_EXPERIMENT_NODE"]
            result = {"run_path": str(tmp_path / "runs" / env["OVLAB_EXPERIMENT_ID"] / "evaluate" / node / f"{node}-run")}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=fake_runner)
    result = workflow.run(_write(tmp_path, spec), sequential=True)
    assert maximum == 1
    assert result["execution"] == {"mode": "sequential", "max_workers": 1}
    assert workflow.plan(_write(tmp_path, spec), max_workers=1)["execution"]["mode"] == "sequential"


def test_parallel_failure_preserves_completed_run_and_blocks_dependents(tmp_path):
    spec = {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "parallel-failure", "name": "Failure",
        "config": {"train": {"profile": "openvla-lora-libero10-smoke.yaml"},
                   "evaluate": {"profile": "libero10-lora-4bit-three-per-task.yaml"}},
        "children": [
            {"id": "a", "execute": ["train"]},
            {"id": "b", "execute": ["evaluate"]},
            {"id": "c", "execute": ["evaluate"], "depends_on": [{"node": "a", "phase": "train"}]},
        ],
    }
    barrier = threading.Barrier(2, timeout=5)
    seen = []
    workflow = None

    def fake_runner(command, **kwargs):
        env = kwargs["env"]
        node = env["OVLAB_EXPERIMENT_NODE"]
        seen.append((command[1], node))
        if command[1:3] == ["export", "isolated"]:
            barrier.wait()
            return SimpleNamespace(returncode=1, stdout=json.dumps({"status": "error", "errors": [{"message": "report failed"}]}))
        if node == "b":
            barrier.wait()
            assert workflow._cancelled.wait(5)
        category = "train" if command[1] == "train" else "evaluate"
        result = {"host_path": str(tmp_path / "runs" / env["OVLAB_EXPERIMENT_ID"] / category / node / f"{node}-run")}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path)}, runner=fake_runner)
    with pytest.raises(ExperimentExecutionError, match="report failed"):
        workflow.run(_write(tmp_path, spec))
    result_path = next((tmp_path / "runs").glob("*/.experiment/result.json"))
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["nodes"]["a"]["train"]["run_id"] == "a-run"
    assert result["nodes"]["b"]["evaluate"]["run_id"] == "b-run"
    assert not any(node == "c" for _, node in seen)
    assert result["failures"][0]["node"] == "a"


def test_cancelling_experiment_interrupts_active_subprocess(tmp_path):
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    launcher = tmp_path / "ovlab"
    launcher.write_text(
        "#!/usr/bin/env python3\nimport json,signal,sys,time\nfrom pathlib import Path\n"
        f"ready=Path({str(ready)!r}); stopped=Path({str(stopped)!r})\n"
        "def stop(*args):\n stopped.touch(); print(json.dumps({'status':'error','errors':[{'message':'interrupted'}]}),flush=True); sys.exit(130)\n"
        "signal.signal(signal.SIGINT,stop)\nready.touch()\nwhile True: time.sleep(.05)\n"
    )
    launcher.chmod(0o755)
    workflow = ExperimentWorkflow(ROOT)
    workflow.repository_root = tmp_path
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(workflow._invoke, ["fake"], dict(os.environ))
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        workflow._cancel_running()
        with pytest.raises(ExperimentExecutionError, match="interrupted"):
            future.result(timeout=5)
    assert stopped.exists()


@pytest.mark.parametrize("value", [0, -1, 33, True])
def test_invalid_parallel_limit_fails_before_writing(tmp_path, value):
    workflow = ExperimentWorkflow(ROOT, environment={"OVLAB_DATA_ROOT": str(tmp_path / "data")})
    with pytest.raises(ExperimentSpecError, match="max workers"):
        workflow.run(_write(tmp_path, _spec()), max_workers=value)
    assert not (tmp_path / "data").exists()


def test_dry_run_writes_nothing(tmp_path):
    spec = _spec()
    env = {"OVLAB_RUNS_ROOT": str(tmp_path / "runs"), "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports")}
    result = ExperimentWorkflow(ROOT, environment=env).run(_write(tmp_path, spec), dry_run=True)
    assert result["side_effects_performed"] is False
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "exports").exists()


def test_run_executes_train_then_eval_with_valid_materialized_profiles(tmp_path):
    spec = {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "train-eval-tree", "name": "Train and evaluate",
        "config": {
            "train": {"profile": "openvla-lora-libero10-smoke.yaml"},
            "evaluate": {"profile": "libero10-lora-4bit-three-per-task.yaml"},
        },
        "children": [
            {"id": "model-a", "execute": ["train", "evaluate"], "config": {"evaluate": {
                "component_patches": {"benchmark": {"settings": {"task_indices": [0]}}}
            }}},
            {"id": "model-b", "execute": ["train", "evaluate"], "depends_on": [
                {"node": "model-a", "phase": "evaluate"}
            ]},
        ],
    }
    seen = []

    def fake_runner(command, **kwargs):
        env = kwargs["env"]
        instance, node = env["OVLAB_EXPERIMENT_ID"], env["OVLAB_EXPERIMENT_NODE"]
        category = command[1]
        if category == "train":
            assert load_training_profile(command[3], config_root=ROOT / "experiments").document["training"]["seed"] == 7
            result = {"host_path": str(tmp_path / "runs" / instance / "train" / node / f"trained-{node}")}
        elif category == "eval":
            generated = Path(command[3])
            relative = generated.relative_to(ROOT).as_posix()
            ConfigBundleBuilder(ROOT).build(relative, None)
            result = {"run_path": str(tmp_path / "runs" / instance / "evaluate" / node / f"evaluated-{node}")}
        else:
            assert category == "export"
            export_kind = command[2]
            export_category = command[command.index("--category") + 1]
            result = {"output": str(tmp_path / "exports" / instance / export_category / node / export_kind)}
            category = f"export-{export_category}-{export_kind}"
        seen.append((category, node))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    env = {"OVLAB_RUNS_ROOT": str(tmp_path / "runs"), "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports")}
    result = ExperimentWorkflow(ROOT, environment=env, runner=fake_runner).run(_write(tmp_path, spec), sequential=True)
    assert seen == [
        ("train", "model-a"), ("export-train-isolated", "model-a"), ("eval", "model-a"),
        ("train", "model-b"), ("export-train-isolated", "model-b"), ("eval", "model-b"),
        ("export-train-grouped", ""), ("export-evaluate-grouped", ""),
    ]
    assert result["nodes"]["model-a"]["train"]["run_id"] == "trained-model-a"
    assert result["nodes"]["model-a"]["evaluate"]["run_id"] == "evaluated-model-a"
    assert result["grouped"]["train"][""]["status"] == "completed"
    assert result["grouped"]["evaluate"][""]["status"] == "completed"
    assert result["nodes"]["model-a"]["train"]["isolated_export"]["output"].endswith("/train/model-a/isolated")


def test_host_evaluation_publishes_report_after_run(tmp_path):
    spec = {
        "schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "host-eval", "name": "Host evaluation",
        "config": {"evaluate": {"profile": "mock-e2e-smoke.yaml"}},
        "execute": ["evaluate"],
    }
    seen = []

    def fake_runner(command, **kwargs):
        seen.append(command[1:3])
        instance = kwargs["env"]["OVLAB_EXPERIMENT_ID"]
        result = ({"run_path": str(tmp_path / "runs" / instance / "evaluate" / "mock-run")}
                  if command[1] == "run" else {"status": "completed", "report": {"interactive_output": "report.html"}})
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "result": result}))

    env = {"OVLAB_RUNS_ROOT": str(tmp_path / "runs"), "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports")}
    result = ExperimentWorkflow(ROOT, environment=env, runner=fake_runner).run(_write(tmp_path, spec))
    assert [step[0] for step in seen] == ["run", "report"]
    assert result["nodes"][""]["evaluate"]["postprocessing"]["report"]["interactive_output"] == "report.html"


def test_run_rejects_selection_with_no_executable_phase(tmp_path):
    spec = _spec()
    env = {"OVLAB_RUNS_ROOT": str(tmp_path / "runs"), "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports")}
    with pytest.raises(ExperimentSpecError, match="no executable"):
        ExperimentWorkflow(ROOT, environment=env).run(_write(tmp_path, spec), node_only=True)
    assert not (tmp_path / "runs").exists()


def test_experiment_index_escapes_labels_and_links_only_inside_export_root(tmp_path):
    spec = _spec()
    spec["name"] = '<img src=x onerror="alert(1)">'
    spec["children"][0]["name"] = "<script>unsafe</script>"
    plan = ExperimentWorkflow(ROOT).plan(_write(tmp_path, spec))
    output = tmp_path / "exports" / "instance"
    report = output / "evaluate" / "model-a" / "interactive" / "one" / "libero-task-default" / "build"
    report.mkdir(parents=True)
    (report / "index.html").write_text("safe")
    (report.parent / "latest.json").write_text(json.dumps({"build_id": "build"}))
    result = {"experiment_id": "instance", "status": "completed", "nodes": {
        "model-a": {"evaluate": {"run_id": "one", "run_path": "/outside/one", "result": {
            "interactive_output": "/etc/passwd",
        }}}
    }, "grouped": {}}
    markup = build_experiment_index(plan, result, output)
    assert "<script>unsafe</script>" not in markup
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in markup
    assert "<img src=x" not in markup
    assert "evaluate/model-a/interactive/one/libero-task-default/build/index.html" in markup
    assert "/etc/passwd" not in markup


def test_experiment_index_links_to_isolated_html_when_available(tmp_path):
    plan = ExperimentWorkflow(ROOT).plan(_write(tmp_path, _spec()))
    output = tmp_path / "exports" / "instance"
    isolated = output / "evaluate" / "model-a" / "isolated" / "one"
    isolated.mkdir(parents=True)
    (isolated / "metadata.json").write_text("{}")
    (isolated / "index.html").write_text("<h1>Isolated</h1>")
    result = {"experiment_id": "instance", "status": "completed", "nodes": {
        "model-a": {"evaluate": {"run_id": "one", "run_path": "/outside/one", "result": {}}}
    }, "grouped": {}}
    markup = build_experiment_index(plan, result, output)
    assert 'href="evaluate/model-a/isolated/one/index.html"' in markup
    assert 'href="evaluate/model-a/isolated/one/metadata.json"' not in markup


def test_training_export_cli_infers_scope_from_selected_run(tmp_path, monkeypatch):
    from ovlab_benchctl import cli
    import ovlab_runner.training_exports as training_exports

    runs = tmp_path / "runs"
    exports = tmp_path / "exports"
    instance = "model-study_2026-11-09_abcdef01"
    selected = {
        "first": runs / instance / "train" / "model" / "task-a" / "first",
        "second": runs / instance / "train" / "model" / "task-b" / "second",
    }
    calls = []

    class App:
        def _data_roots(self):
            return runs, exports / "evaluate" / "derived", exports

        def _training_run(self, reference):
            return SimpleNamespace(run_id=reference, path=selected[reference])

    class Engine:
        def __init__(self, _runs, _exports, *, scope=None):
            calls.append(("scope", scope.experiment_id if scope else None, scope.node_path if scope else None))

        def generate_isolated(self, run_id):
            return {"run_id": run_id}

        def generate_grouped(self, group_name, *, run_ids):
            return {"name": group_name, "run_ids": run_ids}

        def verify(self, kind, name):
            return {"kind": kind, "name": name}

    monkeypatch.setattr(cli, "_application", lambda: App())
    monkeypatch.setattr(training_exports, "TrainingExportEngine", Engine)
    monkeypatch.delenv("OVLAB_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("OVLAB_EXPERIMENT_NODE", raising=False)
    args = cli._parser().parse_args(["export", "isolated", "--category", "train", "--run", "first"])
    assert cli._dispatch(args)[0]["run_id"] == "first"
    assert calls[-1] == ("scope", instance, "model/task-a")
    args = cli._parser().parse_args(["export", "grouped", "--category", "train", "--name", "tasks", "--runs", "first", "second"])
    assert cli._dispatch(args)[0]["run_ids"] == ["first", "second"]
    assert calls[-1] == ("scope", instance, "model")
    args = cli._parser().parse_args(["export", "verify", "--category", "train", "--kind", "isolated", "--name", "second"])
    assert cli._dispatch(args)[0]["kind"] == "isolated"
    assert calls[-1] == ("scope", instance, "model/task-b")
    metadata = exports / instance / "train" / "model" / "grouped" / "tasks" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    args = cli._parser().parse_args(["export", "verify", "--category", "train", "--kind", "grouped", "--name", "tasks"])
    assert cli._dispatch(args)[0]["kind"] == "grouped"
    assert calls[-1] == ("scope", instance, "model")
