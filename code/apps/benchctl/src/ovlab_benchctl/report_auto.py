"""Select and run the existing export commands for a run or experiment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess

from .run_references import (
    RunReferenceAmbiguousError, RunReferenceUnavailableError,
    resolve_run_reference_across, resolve_run_reference_tree,
)


class ReportSelectionError(ValueError):
    """The report source cannot be identified unambiguously."""


def _experiment(runs: Path, reference: str):
    candidate = Path(reference).expanduser()
    if candidate.is_absolute():
        directory = candidate
    elif candidate.parts == (reference,) and reference not in {".", ".."}:
        directory = runs / reference
    else:
        return None
    if directory.is_symlink() or not directory.resolve().is_relative_to(runs):
        return None
    if directory.resolve().parent != runs:
        return None
    manifest = directory / ".experiment" / "result.json"
    if not manifest.is_file() or manifest.is_symlink() or manifest.parent.is_symlink():
        return None
    try:
        result = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportSelectionError(f"cannot read experiment result: {manifest}") from exc
    if result.get("experiment_id") != directory.name or not isinstance(result.get("nodes"), dict):
        raise ReportSelectionError(f"invalid experiment result: {manifest}")
    if result.get("status") != "completed":
        raise ReportSelectionError(f"experiment {directory.name!r} is not completed ({result.get('status', 'unknown')})")
    plan_path = directory / ".experiment" / "plan.json"
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ReportSelectionError(f"experiment plan is unavailable: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportSelectionError(f"cannot read experiment plan: {plan_path}") from exc
    if (plan.get("experiment_id") != directory.name or
            not isinstance(plan.get("nodes"), list) or not isinstance(plan.get("id"), str)):
        raise ReportSelectionError(f"invalid experiment plan: {plan_path}")
    nodes = {}
    for node in plan["nodes"]:
        if not isinstance(node, dict):
            raise ReportSelectionError("invalid experiment plan node")
        path, node_id, children = node.get("path"), node.get("id"), node.get("children")
        if (not isinstance(path, str) or not isinstance(node_id, str) or
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node_id) or
                not isinstance(children, list) or path in nodes or
                any(not isinstance(child, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", child) for child in children) or
                len(set(children)) != len(children)):
            raise ReportSelectionError("invalid experiment plan node")
        if path and (any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in path.split("/")) or path.split("/")[-1] != node_id):
            raise ReportSelectionError("invalid experiment node path")
        nodes[path] = node
    if "" not in nodes or nodes[""]["id"] != plan["id"]:
        raise ReportSelectionError("experiment plan has no valid root")
    for path, node in nodes.items():
        for child in node["children"]:
            if f"{path}/{child}".strip("/") not in nodes:
                raise ReportSelectionError("experiment plan references an unknown child")
        if path:
            parent, _, child = path.rpartition("/")
            if parent not in nodes or child not in nodes[parent]["children"]:
                raise ReportSelectionError("experiment plan contains an orphan node")
    groups = {"train": {}, "evaluate": {}}
    for path, node in result["nodes"].items():
        if path not in nodes or not isinstance(node, dict):
            raise ReportSelectionError("invalid experiment node in result")
        for category in groups:
            record = node.get(category)
            if record is None:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
                raise ReportSelectionError(f"invalid {category} run in experiment result")
            run_id = record["run_id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
                raise ReportSelectionError(f"invalid {category} run ID in experiment result")
            if run_id in groups[category].values():
                raise ReportSelectionError(f"duplicate {category} run ID in experiment result: {run_id}")
            groups[category][path] = run_id
    if not any(groups.values()):
        raise ReportSelectionError(f"experiment {directory.name!r} contains no completed runs")
    jobs = []
    for category, completed in groups.items():
        for path, run_id in completed.items():
            jobs.append({"category": category, "type": "isolated", "runs": [run_id], "node": path})
        # Match workflow grouping: a parent compares runs from at least two
        # represented child branches, independently at every hierarchy level.
        for path, node in reversed(list(nodes.items())):
            child_runs = []
            represented = 0
            for child in node["children"]:
                child_path = f"{path}/{child}".strip("/")
                branch = [run_id for run_path, run_id in completed.items()
                          if run_path == child_path or run_path.startswith(child_path + "/")]
                represented += bool(branch)
                child_runs.extend(branch)
            if represented >= 2 and len(child_runs) >= 2:
                jobs.append({"category": category, "type": "grouped", "runs": child_runs,
                             "node": path, "name": node["id"] + "-comparison"})
    return directory.name, jobs


def _single(runs: Path, model_data: Path, reference: str):
    try:
        found = resolve_run_reference_tree(runs, reference)
        relative = found.path.relative_to(runs).parts
        category = "train" if relative[0] == "train" or len(relative) > 1 and relative[1] == "train" else "evaluate"
        return category, found
    except RunReferenceUnavailableError:
        pass
    candidates = []
    for category, roots in (("evaluate", (runs / "evaluate", runs)),
                            ("train", (runs / "train", model_data / "training-runs"))):
        try:
            found = resolve_run_reference_across(roots, reference)
            if any((found.path / marker).is_file() for marker in (
                "manifest.started.json", "manifest.completed.json", "manifest.failed.json", "training-plan.json",
            )):
                candidates.append((category, found))
        except RunReferenceUnavailableError:
            pass
    if len(candidates) > 1:
        raise RunReferenceAmbiguousError(f"run reference is ambiguous across categories: {reference}")
    if not candidates:
        raise RunReferenceUnavailableError(f"no run or completed experiment matches {reference!r}")
    return candidates[0]


def generate_report(reference: str, *, repository_root: Path, environment=None, runner=None):
    env = dict(os.environ if environment is None else environment)
    data = Path(env.get("OVLAB_DATA_ROOT", repository_root.parent / "ovlab-data")).expanduser().resolve()
    model_data = Path(env.get("OVLAB_MODEL_DATA_ROOT", data)).expanduser().resolve()
    runs = Path(env.get("OVLAB_RUNS_ROOT", data / "runs")).expanduser().resolve()
    experiment = _experiment(runs, reference)
    if experiment:
        experiment_id, jobs = experiment
        source_type, source = "experiment", experiment_id
    else:
        category, run = _single(runs, model_data, reference)
        if not run.path.is_relative_to(runs):
            raise ReportSelectionError("legacy training runs outside the configured runs root cannot be exported automatically")
        jobs = [{"category": category, "type": "isolated", "runs": [run.run_id], "node": None}]
        source_type, source = "run", run.run_id
        experiment_id = None
    invoke = runner or subprocess.run
    exports = []
    for job in jobs:
        category, kind, ids = job["category"], job["type"], job["runs"]
        args = [str(repository_root / "ovlab"), "export", kind, "--category", category]
        if kind == "grouped":
            args += ["--name", job["name"], "--runs", *ids]
        else:
            args += ["--run", ids[0]]
        args.append("--json")
        child_env = dict(env)
        if experiment_id is not None:
            child_env.update(OVLAB_EXPERIMENT_ID=experiment_id, OVLAB_EXPERIMENT_NODE=job["node"])
        else:
            child_env.pop("OVLAB_EXPERIMENT_ID", None)
            child_env.pop("OVLAB_EXPERIMENT_NODE", None)
        completed = invoke(args, cwd=repository_root, env=child_env, stdout=subprocess.PIPE, text=True, check=False)
        try:
            payload = json.loads(completed.stdout)
        except (ValueError, TypeError) as exc:
            raise ReportSelectionError(f"{category} {kind} export returned invalid output") from exc
        if completed.returncode != 0 or payload.get("status") != "success":
            message = "; ".join(str(item.get("message", item)) for item in payload.get("errors", []))
            completed_paths = "; ".join(item["output"] for item in exports)
            previous = f"; already exported: {completed_paths}" if completed_paths else ""
            raise ReportSelectionError(f"{category} {kind} export failed: {message or completed.returncode}{previous}")
        result = payload.get("result")
        if not isinstance(result, dict) or not result.get("output"):
            raise ReportSelectionError(f"{category} {kind} export did not return an output path")
        exports.append({"category": category, "type": kind, "node": job["node"], "runs": ids, "output": result["output"],
                        "integrity": result.get("integrity"), "reused": result.get("reused")})
    return {"schema_version": "ovlab.report-export/v1", "status": "completed", "source_type": source_type,
            "source": source, "exports": exports}
