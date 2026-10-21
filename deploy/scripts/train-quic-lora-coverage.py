#!/usr/bin/env python3
"""Run ordinary OVLAB profiles, gating every deep run on verified smoke runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
SUITES = ("libero10", "libero-object", "libero-spatial", "libero-goal")
PROFILE_ROOT = Path("configs/experiments/training/quic/lora-coverage")


class StudyError(RuntimeError):
    pass


def run_cli(command, **options):
    """Give the ordinary CLI time to stop its training container on interruption."""
    options.pop("check", None)
    # Forward one interrupt to the CLI, which owns Docker cleanup. Keeping it
    # outside the terminal process group avoids duplicate SIGINT delivery.
    with subprocess.Popen(command, start_new_session=True, **options) as process:
        try:
            stdout, _ = process.communicate()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.communicate(timeout=55)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_json(repository, arguments, evidence, *, runner=run_cli):
    command = [str(repository / "ovlab"), *arguments, "--json"]
    print(shlex.join(command), file=sys.stderr, flush=True)
    completed = runner(command, cwd=repository, stdout=subprocess.PIPE, text=True, check=False)
    try:
        envelope = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        if evidence is not None:
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.with_suffix(".stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        raise StudyError(f"command did not return JSON: {shlex.join(command)}") from error
    if evidence is not None:
        write_json(evidence, envelope)
    if completed.returncode or not isinstance(envelope, dict) or envelope.get("status") != "success":
        raise StudyError(f"command failed: {shlex.join(command)}; details: {envelope}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise StudyError(f"command lacks a result object: {shlex.join(command)}")
    return result


def check_training_result(child):
    result = child.get("result", {})
    run_id = child.get("run_id")
    checkpoint_id = result.get("checkpoint_id")
    if not run_id or not checkpoint_id or result.get("status") != "completed":
        raise StudyError("training did not return a completed run and finalized checkpoint")
    for key in ("identity_logit_max_absolute_error", "reload_logit_max_absolute_error"):
        if result.get(key) != 0.0:
            raise StudyError(f"run {run_id} failed {key}=0")
    if result.get("changed_adapter_tensor_count", 0) <= 0:
        raise StudyError(f"run {run_id} did not update adapter tensors")
    validation = result.get("adapter_validation", {})
    loss = validation.get("action_token_cross_entropy")
    if (not validation.get("action_tokens") or isinstance(loss, bool)
            or not isinstance(loss, (int, float)) or not math.isfinite(loss)):
        raise StudyError(f"run {run_id} lacks finite held-out action-token validation")
    return run_id, checkpoint_id


def check_coverage(child):
    root = child.get("host_path")
    if not isinstance(root, str):
        raise StudyError("training lacks its canonical evidence directory")
    try:
        selection = json.loads((Path(root) / "target-selection.json").read_text(encoding="utf-8"))
        coverage = selection["coverage"]
        selected = set(coverage["selected_modules"])
        has_convolution = any(entry["name"] in selected and entry.get("module_type") == "Conv2d"
                              for entry in coverage["matched_modules"])
        complete = (coverage["active_coverage_fraction"] == 1.0
                    and coverage["uncovered_active_module_count"] == 0 and has_convolution)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StudyError("training lacks the required target-coverage evidence; rebuild the QuIC image") from error
    if not complete:
        raise StudyError("training did not cover all active LoRA targets including Conv2d")
    return {key: coverage[key] for key in ("matched_module_count", "selected_module_count",
                                           "excluded_inactive_count", "active_coverage_fraction")}


def run_study(suites, *, output=None, validate_only=False, dry_run=False,
              repository=REPOSITORY, runner=run_cli):
    suites = tuple(dict.fromkeys(suites))
    if not suites or any(suite not in SUITES for suite in suites):
        raise StudyError("select at least one supported LIBERO suite")
    if not validate_only and output is None:
        raise StudyError("--output NEW_DIRECTORY is required for plans and training")
    if output is not None:
        output = Path(output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": "ovlab.quic-coverage-study/v1", "status": "running",
             "mode": "validate" if validate_only else "plan" if dry_run else "train",
             "suites": list(suites), "children": {},
             "deep_initialization": "original dense host, independently of smoke adapters"}

    def record():
        if output is not None:
            write_json(output / "study.json", state)

    def invoke(arguments, relative):
        return command_json(repository, arguments, output / relative if output is not None else None,
                            runner=runner)

    record()
    try:
        # Check every profile before starting expensive work.
        for stage in ("smoke", "deep"):
            for suite in suites:
                profile = str(PROFILE_ROOT / f"{suite}-{stage}.yaml")
                state["active"] = f"validate/{stage}/{suite}"
                record()
                validation = invoke(["train", "validate", "--profile", profile],
                                    Path(stage) / suite / "profile-validation.json")
                if validation.get("valid") is not True:
                    raise StudyError(f"profile validation failed: {profile}")
        if not validate_only:
            # All selected smokes must complete and pass independent verification
            # before the first deep training command can be issued.
            for stage in ("smoke", "deep"):
                for suite in suites:
                    profile = str(PROFILE_ROOT / f"{suite}-{stage}.yaml")
                    key = f"{stage}/{suite}"
                    state["active"] = key
                    record()
                    arguments = ["train", "plan", "--profile", profile] if dry_run else ["train", "run", profile]
                    child = invoke(arguments, Path(stage) / suite / ("plan.json" if dry_run else "run.json"))
                    if dry_run:
                        state["children"][key] = {"status": "planned", "scientific_training_id": child.get("scientific_training_id")}
                    else:
                        run_id, checkpoint_id = check_training_result(child)
                        verified_run = invoke(["train", "verify", run_id], Path(stage) / suite / "run-verification.json")
                        verified_checkpoint = invoke(["checkpoint", "verify", "--checkpoint", checkpoint_id],
                                                     Path(stage) / suite / "checkpoint-verification.json")
                        if (verified_run.get("status") != "verified" or verified_run.get("run_id") != run_id
                                or verified_checkpoint.get("status") != "verified"
                                or verified_checkpoint.get("checkpoint_id") != checkpoint_id):
                            raise StudyError(f"independent verification failed for {run_id}")
                        coverage = check_coverage(child)
                        state["children"][key] = {"status": "verified", "run_id": run_id,
                                                  "checkpoint_id": checkpoint_id, "coverage": coverage}
                    record()
        state.pop("active", None)
        state["status"] = "completed"
        record()
        return state
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["failure"] = str(error) or "interrupted"
        record()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="validate all selected smoke/deep schemas without datasets or CUDA")
    mode.add_argument("--dry-run", action="store_true", help="resolve all selected smoke/deep plans without training")
    parser.add_argument("--suite", choices=SUITES, action="append", help="repeat to select suites; default: all four")
    parser.add_argument("--output", type=Path, help="new directory for study journal and child JSON evidence")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output is None:
        parser.error("--output NEW_DIRECTORY is required unless --validate-only is used")
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run_study(args.suite or SUITES, output=args.output,
                           validate_only=args.validate_only, dry_run=args.dry_run)
    except (StudyError, OSError, KeyboardInterrupt) as error:
        print(f"QuIC coverage study stopped: {error or 'interrupted'}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
