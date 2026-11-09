import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovlab_runner import ArtifactError, TrainingDerivedReportEngine, TrainingInteractiveReportEngine, build_training_performance_model


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_run(root):
    run = root / "training-fixture"
    run.mkdir(parents=True)
    _write_json(run / "training-plan.json", {
        "scientific": {"profile": {"id": "lora-smoke"}},
    })
    _write_json(run / "result.json", {
        "schema_version": "ovlab.training-result/v1", "run_id": run.name,
        "status": "completed", "peak_vram_bytes": 5000,
        "peak_reserved_vram_bytes": 6000,
        "performance": {
            "parameter_counts": {
                "total": 1000, "trainable": 100, "frozen": 900,
                "adapter": 100, "trainable_adapter": 100,
            },
            "estimated_compute_method": "dense-parameter-token-proxy/training-v1",
        },
    })
    rows = [
        {
            "schema_version": "ovlab.training-metric/v1", "global_step": step,
            "training_loss": 1.0 / step, "step_duration_ms": 10.0 * step,
            "gpu_memory_allocated_bytes": 1024 * 1024 * step,
            "gpu_memory_reserved_bytes": 2 * 1024 * 1024 * step,
            "gpu_memory_peak_bytes": 3 * 1024 * 1024 * step,
            "gpu_memory_peak_reserved_bytes": 4 * 1024 * 1024 * step,
            "estimated_gflops": 5.0 * step,
            "performance": {"estimated_compute": {
                "method": "dense-parameter-token-proxy/training-v1",
                "formula": "(2 * runtime_parameters + 4 * trainable_parameters) * non_padding_tokens * forward_backward_passes",
                "qualification": "analytical proxy; not hardware FLOP measurement",
            }},
        }
        for step in (1, 2)
    ]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    files = []
    for path in sorted(run.iterdir()):
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": _sha(path)})
    _write_json(run / "manifest.json", {
        "schema_version": "ovlab.training-run/v1", "run_id": run.name,
        "status": "completed", "scientific_training_id": "training-abc",
        "execution_plan_id": "execution-def", "checkpoint_id": "checkpoint-123",
        "files": files,
    })
    return run


def test_training_performance_report_tracks_vram_compute_and_parameter_classes(tmp_path):
    run = _training_run(tmp_path / "training-runs")
    model = build_training_performance_model(run)
    assert model["parameter_counts"]["adapter"] == 100
    assert model["system_metrics"]["peak_allocated_bytes"] == 5000
    assert model["system_metrics"]["estimated_total_gflops"] == 15.0
    assert model["system_metrics"]["compute_estimator"]["method"].endswith("training-v1")
    assert model["system_metrics"]["statistics"]["allocated_mib"]["maximum"] == 2.0

    engine = TrainingDerivedReportEngine(run.parent, tmp_path / "derived")
    generated = engine.generate(run.name)
    target = tmp_path / "derived/training/training-fixture/system-performance" / generated["derived_build_id"]
    assert generated["integrity"] == "verified"
    assert (target / "assets/vram_tracking.svg").is_file()
    assert (target / "assets/estimated_compute.svg").is_file()
    html = (target / "index.html").read_text(encoding="utf-8")
    assert "Model complexity" in html and "not measured hardware FLOPs" in html
    assert engine.verify(run.name)["canonical_training_run_modified"] is False
    interactive = Path(generated["interactive_output"])
    assert interactive.parent == tmp_path / "interactive/training/training-fixture/system-performance"
    dashboard = json.loads((interactive / "report.json").read_text())
    assert dashboard["evidence"] is None
    assert dashboard["counts"]["examples_seen"] is None
    assert dashboard["counts"]["dataset_records"] is None
    assert generated["interactive_integrity"] == "verified"
    assert engine.generate(run.name)["reused"] is True
    (interactive / "index.html").write_text("tamper")
    with pytest.raises(ArtifactError, match="interactive report file checksum mismatch"):
        engine.verify(run.name)


def test_unified_training_report_exposes_complete_evidence_without_derived_tree(tmp_path):
    run = _training_run(tmp_path / "runs" / "exp-one" / "train" / "model-a" / "ablation-1")
    root = tmp_path / "exports" / "exp-one" / "train" / "model-a" / "ablation-1" / "interactive"
    engine = TrainingInteractiveReportEngine(run.parent, root)
    before = (run / "manifest.json").read_bytes()
    generated = engine.generate(run.name)
    target = Path(generated["output"])
    model = json.loads((target / "model.json").read_text())
    assert model["system_metrics"]["estimated_total_gflops"] == 15.0
    assert model["charts"]["vram_tracking"]
    assert model["source"]["training_manifest_sha256"] == _sha(run / "manifest.json")
    assert "Complete report evidence" in (target / "index.html").read_text()
    assert not list((tmp_path / "exports").rglob("derived"))
    assert engine.verify(run.name)["integrity"] == "verified"
    assert engine.generate(run.name)["reused"] is True
    assert before == (run / "manifest.json").read_bytes()


def test_training_report_uses_json_parameter_configuration(tmp_path):
    from ovlab_benchctl.application import OvlabApplication
    from ovlab_runner.key_parameters import default_key_parameters

    run = _training_run(tmp_path / "runs" / "exp-one" / "train" / "model-a")
    parameters = default_key_parameters("train")
    group = next(group for group in parameters["groups"] if group["stage"] == "policy")
    group["items"] = [next(item for item in group["items"] if item["label"] == "Trainable parameters")]
    group["items"][0]["label"] = "Selected weights"
    parameters["groups"] = [group]
    config = tmp_path / "training-parameters.json"
    config.write_text(json.dumps(parameters), encoding="utf-8")
    app = OvlabApplication(Path(__file__).resolve().parents[4], environment={"OVLAB_DATA_ROOT": str(tmp_path)})
    default = app.train_report(run.name)
    generated = app.train_report(run.name, parameters=str(config))
    assert generated["interactive_build_id"] != default["interactive_build_id"]
    model = json.loads((Path(generated["output"]) / "model.json").read_text())
    assert model["key_parameters"][0]["items"] == [{
        "label": "Selected weights", "value": 100,
        "source": "run/result.json:performance.parameter_counts.trainable",
    }]
    assert app.train_report(run.name, verify=True, parameters=str(config))["integrity"] == "verified"
    legacy = TrainingDerivedReportEngine(run.parent, tmp_path / "legacy-derived", key_parameters=parameters)
    legacy_report = legacy.generate(run.name)
    assert "Selected weights" in (Path(legacy_report["output"]) / "index.html").read_text()
    assert legacy.verify(run.name)["integrity"] == "verified"


def test_training_application_routes_scoped_run_to_interactive_only(tmp_path):
    from ovlab_benchctl.application import OvlabApplication

    run = _training_run(tmp_path / "runs" / "exp-one" / "train" / "model-a")
    repository = Path(__file__).resolve().parents[4]
    app = OvlabApplication(repository, environment={"OVLAB_DATA_ROOT": str(tmp_path)})
    generated = app.train_report(run.name)
    assert Path(generated["output"]).is_relative_to(
        tmp_path / "exports" / "exp-one" / "train" / "model-a" / "interactive"
    )
    assert not list((tmp_path / "exports").rglob("derived"))
    assert app.train_report(run.name, verify=True)["interactive_build_id"] == generated["interactive_build_id"]


def test_training_application_writes_typed_exports_and_verifies_legacy_reports(tmp_path, monkeypatch):
    from ovlab_benchctl.application import OvlabApplication

    run = _training_run(tmp_path / "runs/train")
    repository = Path(__file__).resolve().parents[4]
    app = OvlabApplication(repository, environment={"OVLAB_DATA_ROOT": str(tmp_path)})
    generated = app.train_report(run.name)
    assert Path(generated["output"]).is_relative_to(tmp_path / "exports/train/derived" / run.name)
    assert Path(generated["interactive_output"]).is_relative_to(tmp_path / "exports/train/interactive" / run.name)
    assert app.train_report(run.name, verify=True)["integrity"] == "verified"
    legacy = TrainingDerivedReportEngine(run.parent, tmp_path / "exports/derived")
    legacy.generate(run.name)
    Path(generated["output"]).rename(Path(generated["output"]).with_name("newer-build"))
    assert app.train_report(run.name, verify=True, build_id=generated["derived_build_id"])["integrity"] == "verified"
    (tmp_path / "exports/train").rename(tmp_path / "saved-typed-exports")
    monkeypatch.setenv("OVLAB_DERIVED_DISPLAY_ROOT", str(tmp_path / "exports/train/derived"))
    verified = app.train_report(run.name, verify=True)
    assert verified["integrity"] == "verified"
    assert Path(verified["interactive_output"]).is_dir()


def test_new_training_build_does_not_change_previous_build_permissions(tmp_path, monkeypatch):
    import ovlab_runner.training_derived as training_module

    run = _training_run(tmp_path / "runs/train")
    engine = TrainingDerivedReportEngine(run.parent, tmp_path / "exports/train/derived")
    monkeypatch.setattr(training_module, "TRAINING_REPORT_RENDERER", "fixture-v1")
    old = Path(engine.generate(run.name)["output"])
    old.chmod(0o550)
    try:
        monkeypatch.setattr(training_module, "TRAINING_REPORT_RENDERER", "fixture-v2")
        new = engine.generate(run.name)
        assert Path(new["output"]) != old
        assert old.stat().st_mode & 0o7777 == 0o550
        assert (old.parent / "latest.json").stat().st_mode & 0o777 == 0o640
    finally:
        old.chmod(0o2770)


def test_interactive_training_uses_actual_counts_and_forward_pass_evidence(tmp_path):
    from ovlab_benchctl.training_evidence import action_token_counts, capture_data_flow
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")

    batch = {"input_ids": torch.tensor([[1, 10, 11, 2]]), "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[-100, 10, 11, 2]]), "pixel_values": torch.zeros((1, 6, 64, 64))}
    # Two vision tokens, then aligned text logits. EOS is excluded from action accuracy.
    logits = torch.zeros((1, 6, 16))
    logits[0, 2, 10] = 1
    logits[0, 3, 12] = 1
    logits[0, 4, 2] = 1
    output = SimpleNamespace(logits=logits, loss=torch.tensor(.75))
    assert action_token_counts(output, batch["labels"], 9) == (1, 2)
    tokenizer = SimpleNamespace(decode=lambda ids: "decoded " + str(ids))
    evidence = capture_data_flow(batch, output, tokenizer, dataset_records=123,
        pixel_dtype=torch.bfloat16, optimizer=SimpleNamespace(defaults={"lr": .001, "betas": (.9, .999)}))
    assert evidence["input_batch"]["pixel_values"]["shape"] == [1, 6, 64, 64]
    assert evidence["input_batch"]["pixel_values"]["dtype"] == "torch.bfloat16"
    assert evidence["input_record"]["pixel_values"]["dtype"] == "torch.bfloat16"
    assert evidence["input_record"]["pixel_values"]["truncated"] is True
    assert evidence["output_record"]["predicted_tokens"]["values"] == [10, 12, 2]
    assert evidence["output_batch"]["logits"]["shape"] == [1, 6, 16]
    run = _training_run(tmp_path / "training-runs")
    _write_json(run / "data-flow.json", evidence)
    plan = json.loads((run / "training-plan.json").read_text())
    plan["scientific"]["training"] = {"per_device_batch_size": 1, "gradient_accumulation_steps": 2, "learning_rate": .001}
    _write_json(run / "training-plan.json", plan)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        row.update(examples_seen=2, microbatches=2, action_token_count=4, action_token_correct=2, action_token_accuracy=.5)
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"] = [{"path": path.name, "size": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(run.iterdir()) if path.name != "manifest.json"]
    _write_json(run / "manifest.json", manifest)
    before = {path.name: _sha(path) for path in run.iterdir()}
    generated = TrainingDerivedReportEngine(run.parent, tmp_path / "derived").generate(run.name)
    dashboard = json.loads((Path(generated["interactive_output"]) / "report.json").read_text())
    parameters = {item["label"]: item for group in dashboard["key_parameters"] for item in group["items"]}
    assert parameters["Learning rate"] == {
        "label": "Learning rate", "value": .001,
        "source": "run/training-plan.json:scientific.training.learning_rate",
    }
    assert parameters["Batch size per device"]["value"] == 1
    assert parameters["Gradient accumulation steps"]["value"] == 2
    assert parameters["Training objective"]["value"] is None
    derived_html = (Path(generated["output"]) / "index.html").read_text()
    assert "Key parameters" in derived_html and "0.001" in derived_html
    assert "run/training-plan.json:scientific.training.learning_rate" in derived_html
    assert dashboard["counts"] == {"optimizer_steps": 2, "dataset_records": 123, "examples_seen": 4,
        "microbatches": 4, "supervised_action_tokens": 8}
    assert dashboard["hyperparameters"]["optimizer"]["betas"] == [.9, .999]
    assert dashboard["evidence"]["output_record"]["targets"]["values"] == [10, 11, 2]
    assert {path.name: _sha(path) for path in run.iterdir()} == before


def test_training_report_rejects_unfinalized_batch_evidence(tmp_path):
    run = _training_run(tmp_path / "training-runs")
    _write_json(run / "data-flow.json", {"dataset_records": 999})
    with pytest.raises(ArtifactError, match="absent from the finalized inventory"):
        TrainingDerivedReportEngine(run.parent, tmp_path / "derived").generate(run.name)


def test_training_performance_report_cli_uses_read_only_source_contract(tmp_path):
    model_data = tmp_path / "model-data"
    run = _training_run(model_data / "training-runs")
    repository = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [str(repository / "ovlab"), "train", "report", "--run", run.name, "--json"],
        cwd=repository,
        env={
            **os.environ, "OVLAB_PYTHON": sys.executable,
            "OVLAB_REPORTING_RUNTIME": "host",
            "OVLAB_MODEL_DATA_ROOT": str(model_data),
            "OVLAB_EXPORTS_ROOT": str(tmp_path / "exports"),
        },
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0 and completed.stderr == ""
    result = json.loads(completed.stdout)["result"]
    assert result["training_run_id"] == run.name
    assert result["canonical_training_run_modified"] is False
