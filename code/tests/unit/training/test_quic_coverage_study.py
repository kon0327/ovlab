"""The coverage experiment must never start deep training after a failed smoke."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[4] / "deploy/scripts/train-quic-lora-coverage.py"
spec = importlib.util.spec_from_file_location("quic_coverage_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fake_cli(tmp_path, *, fail_smoke=None, fail_verify=False, missing_conv=False):
    calls = []

    def runner(command, **kwargs):
        arguments = command[1:-1]
        calls.append(arguments)
        exit_code = 0
        result = {"valid": True}
        if arguments[:2] == ["train", "run"]:
            name = Path(arguments[2]).stem
            if name == fail_smoke:
                return subprocess.CompletedProcess(command, 1, json.dumps({"status": "error", "errors": ["CUDA OOM"]}))
            root = tmp_path / "runs" / name
            root.mkdir(parents=True)
            (root / "target-selection.json").write_text(json.dumps({"coverage": {
                "active_coverage_fraction": 1.0, "uncovered_active_module_count": 0,
                "selected_modules": ["vision.patch_embed.proj"],
                "matched_modules": [{"name": "vision.patch_embed.proj",
                                     "module_type": "Linear" if missing_conv else "Conv2d"}],
                "matched_module_count": 2, "selected_module_count": 1, "excluded_inactive_count": 1,
            }}))
            result = {"run_id": name, "host_path": str(root), "result": {
                "status": "completed", "checkpoint_id": f"checkpoint-{name}",
                "identity_logit_max_absolute_error": 0.0, "reload_logit_max_absolute_error": 0.0,
                "changed_adapter_tensor_count": 1,
                "adapter_validation": {"action_tokens": 28, "action_token_cross_entropy": 1.0},
            }}
        elif arguments[:2] == ["train", "verify"]:
            result = {"status": "failed" if fail_verify else "verified", "run_id": arguments[2]}
        elif arguments[:2] == ["checkpoint", "verify"]:
            result = {"status": "verified", "checkpoint_id": arguments[3]}
        elif arguments[:2] == ["train", "plan"]:
            result = {"scientific_training_id": "scientific-fixture"}
        return subprocess.CompletedProcess(command, exit_code, json.dumps({"status": "success", "result": result}))

    return calls, runner


def test_all_smokes_verified_before_first_deep_and_evidence_preserved(tmp_path):
    calls, runner = fake_cli(tmp_path)
    output = tmp_path / "study"
    result = study.run_study(study.SUITES, output=output, runner=runner)
    runs = [Path(call[2]).stem for call in calls if call[:2] == ["train", "run"]]
    assert runs == [f"{suite}-smoke" for suite in study.SUITES] + [f"{suite}-deep" for suite in study.SUITES]
    first_deep = next(index for index, call in enumerate(calls) if call[:2] == ["train", "run"] and "deep" in call[2])
    assert sum(call[:2] == ["train", "verify"] for call in calls[:first_deep]) == 4
    assert sum(call[:2] == ["checkpoint", "verify"] for call in calls[:first_deep]) == 4
    assert len(result["children"]) == 8
    assert json.loads((output / "study.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        study.run_study(study.SUITES, output=output, runner=runner)


@pytest.mark.parametrize("failure", ["trainer", "verification", "coverage"])
def test_smoke_failure_blocks_every_deep_and_keeps_failure_evidence(tmp_path, failure):
    calls, runner = fake_cli(tmp_path, fail_smoke="libero-object-smoke" if failure == "trainer" else None,
                             fail_verify=failure == "verification", missing_conv=failure == "coverage")
    output = tmp_path / "study"
    with pytest.raises(study.StudyError):
        study.run_study(study.SUITES, output=output, runner=runner)
    assert not any(call[:2] == ["train", "run"] and "deep" in call[2] for call in calls)
    result = json.loads((output / "study.json").read_text())
    assert result["status"] == "failed"
    assert result["active"].startswith("smoke/")
    assert result["failure"]


@pytest.mark.parametrize("validate_only,dry_run", [(True, False), (False, True)])
def test_readonly_modes_never_start_training(tmp_path, validate_only, dry_run):
    calls, runner = fake_cli(tmp_path)
    result = study.run_study(["libero-goal"], output=tmp_path / "study",
                             validate_only=validate_only, dry_run=dry_run, runner=runner)
    assert result["status"] == "completed"
    assert all(call[:2] in (["train", "validate"], ["train", "plan"]) for call in calls)
    assert len(calls) == (2 if validate_only else 4)
