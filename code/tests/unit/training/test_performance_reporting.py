import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovlab_runner import ArtifactError, TrainingDerivedReportEngine, build_training_performance_model


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
