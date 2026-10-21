"""Optional host-side W&B mirror of finalized OVLAB runs.

The isolated training and evaluation containers do not need network access or
W&B credentials. Canonical files remain the source of truth.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _metric_values(path: Path) -> dict[str, int | float]:
    values = {}
    for metric in _read_json(path):
        if metric.get("status") != "available":
            continue
        value = _number(metric.get("value"))
        if value is not None:
            values[str(metric["metric_id"])] = value
        elif isinstance(metric.get("value"), dict):
            for name, item in metric["value"].items():
                number = _number(item)
                if number is not None:
                    values[f"{metric['metric_id']}/{name}"] = number
    return values


def _training(run, root: Path) -> None:
    result_path = root / "result.json"
    result = _read_json(result_path) if result_path.is_file() else {}
    run.summary["ovlab/status"] = result.get("status", "unknown")
    if result.get("checkpoint_id"):
        run.summary["ovlab/checkpoint_id"] = result["checkpoint_id"]
    with (root / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("global_step")
            if not isinstance(step, int) or step < 0:
                continue
            data = {"train/global_step": step}
            for key in (
                "training_loss", "learning_rate", "gradient_norm",
                "action_token_accuracy", "examples_per_second", "step_duration_ms",
                "gpu_memory_allocated_bytes", "gpu_memory_reserved_bytes",
                "gpu_memory_peak_bytes", "gpu_memory_peak_reserved_bytes",
                "estimated_gflops", "examples_seen", "non_padding_tokens",
            ):
                value = _number(row.get(key))
                if value is not None:
                    data[f"train/{key}"] = value
            run.log(data, step=step, commit=True)


def _evaluation(run, root: Path) -> None:
    final = root / "manifest.completed.json"
    if not final.is_file():
        final = root / "manifest.failed.json"
    manifest = _read_json(final)
    run.summary["ovlab/status"] = manifest.get("status", "unknown")
    for key in ("task_count", "episode_count", "metric_error_count"):
        value = _number(manifest.get(key))
        if value is not None:
            run.summary[f"eval/{key}"] = value

    totals: dict[str, list[float]] = {}
    episode_index = 0
    tasks_root = root / "tasks"
    tasks = sorted(tasks_root.iterdir()) if tasks_root.is_dir() else ()
    for task in tasks:
        if not task.is_dir():
            continue
        episodes = task / "episodes"
        if episodes.is_dir():
            for episode in sorted(episodes.iterdir()):
                metrics = episode / "metrics.episode.json"
                if not metrics.is_file():
                    continue
                episode_index += 1
                data = {"eval/episode_index": episode_index, "eval/task_id": task.name}
                data.update({f"eval/episode/{key}": value for key, value in _metric_values(metrics).items()})
                run.log(data)
        metrics = task / "metrics.task.json"
        if metrics.is_file():
            for key, value in _metric_values(metrics).items():
                run.summary[f"eval/task/{task.name}/{key}"] = value
                totals.setdefault(key, []).append(float(value))
    for key, values in totals.items():
        run.summary[f"eval/mean/{key}"] = sum(values) / len(values)


def track_run(kind: str, root: str | Path, environment: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """Mirror one finished run when OVLAB_WANDB_PROJECT or WANDB_PROJECT is set.

    Tracking errors are reported separately and never change canonical status.
    """
    env = os.environ if environment is None else environment
    project = (env.get("OVLAB_WANDB_PROJECT") or env.get("WANDB_PROJECT") or "").strip()
    if not project:
        return None
    root = Path(root)
    try:
        import wandb

        if kind == "train":
            plan = _read_json(root / "training-plan.json")
            config = {"ovlab_run_id": root.name, "ovlab_kind": kind, "scientific": plan["scientific"]}
            files = ("training-plan.json", "result.json", "manifest.json")
        elif kind == "eval":
            config = {"ovlab_run_id": root.name, "ovlab_kind": kind, "plan": _read_json(root / "plan.json")}
            files = ("plan.json", "manifest.completed.json", "manifest.failed.json")
        else:
            raise ValueError(f"unsupported W&B run kind: {kind}")

        with wandb.init(
            project=project,
            entity=env.get("OVLAB_WANDB_ENTITY") or env.get("WANDB_ENTITY") or None,
            name=root.name,
            job_type=kind,
            config=config,
        ) as run:
            if kind == "train":
                _training(run, root)
            else:
                _evaluation(run, root)
            artifact = wandb.Artifact(name=f"ovlab-{kind}-{root.name}", type="ovlab-run")
            for name in files:
                path = root / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            run.log_artifact(artifact)
            url = getattr(run, "url", None)
        return {"status": "synced", "url": str(url or "")}
    except Exception as exc:
        print(f"ovlab: W&B tracking failed for {kind} run {root.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
