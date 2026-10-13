"""Offline dashboard companions with the same hierarchy as derived HTML reports."""

from __future__ import annotations

import base64
import hashlib
import html
import importlib.resources
import json
import math
import os
from pathlib import Path
import shutil
import struct
import uuid
import zlib

import numpy as np

from .errors import ArtifactError
from .permissions import finalize_managed_tree


SCHEMA = "ovlab.interactive-report/v1"
RENDERER = "ovlab-interactive-dashboard@1.0.0"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bytes(value):
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read interactive report source: {path}") from exc


def _assets():
    root = importlib.resources.files("ovlab_runner").joinpath("report_assets/static")
    return {name: root.joinpath(name).read_bytes() for name in ("dashboard.css", "dashboard.js")}


def _identity(report):
    manifest = _read(report / "manifest.json")
    return {
        "schema_version": SCHEMA, "renderer": RENDERER,
        "source_report_sha256": _sha(report / "report.json"),
        "source_build_id": manifest["derived_build_id"],
        "source_identity": manifest["identity_inputs"],
        "assets": {name: hashlib.sha256(data).hexdigest() for name, data in _assets().items()},
    }


def dashboard_path(report: Path, derived_root: Path) -> Path:
    build_id = hashlib.sha256(_bytes(_identity(report))).hexdigest()[:20]
    root = derived_root.parent / "interactive"
    target = root / report.parent.relative_to(derived_root) / build_id
    try:
        inside = target.resolve().is_relative_to(root.resolve())
    except OSError:
        inside = False
    if not inside:
        raise ArtifactError("interactive report path escapes the interactive root")
    return target


def _array(root, reference):
    relative = Path(reference["$array"])
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ArtifactError("interactive array reference escapes its episode")
    if _sha(target) != reference["sha256"]:
        raise ArtifactError("interactive array checksum mismatch")
    value = np.load(target, allow_pickle=False)
    if list(value.shape) != reference["shape"] or str(value.dtype) != reference["dtype"]:
        raise ArtifactError("interactive array shape or dtype mismatch")
    return value


def _png(array):
    """Encode a bounded nearest-neighbour RGB preview without image dependencies."""
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        return None
    stride = max(1, math.ceil(max(array.shape[:2]) / 192))
    pixels = np.ascontiguousarray(array[::stride, ::stride])
    height, width = pixels.shape[:2]

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)

    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _eval_episode(source, episode):
    root = source / episode["canonical_episode_path"]
    trace = _read(root / "trace.json")
    observations = {row["step_id"]: row for row in trace["observations"]}
    applied = {}
    for row in trace["executed_actions"]:
        applied.setdefault(row["prediction_id"], []).append(row)
    records = []
    for index, prediction in enumerate(trace["policy_predictions"]):
        observation = observations.get(prediction["step_id"])
        inputs = None
        if observation is not None:
            images = []
            for item in observation["images"]:
                reference = item["data"]
                images.append({
                    "name": item["name"], "shape": reference["shape"], "dtype": reference["dtype"],
                    "encoding": item["encoding"], "color_space": item["color_space"],
                    "preview": _png(_array(root, reference)) if index < 3 and item["color_space"] == "rgb" else None,
                    "source": reference["$array"],
                })
            inputs = {
                "step_id": observation["step_id"], "instruction": observation["instruction"]["text"],
                "images": images,
                "proprioception": [{"name": item["name"], "units": item["units"],
                    "values": _array(root, item["values"]).tolist(), "shape": item["values"]["shape"],
                    "dtype": item["values"]["dtype"]} for item in observation["proprioception"]],
            }
        actions = applied.get(prediction["prediction_id"], [])
        records.append({
            "index": index, "input": inputs,
            "output": {"prediction_id": prediction["prediction_id"], "step_id": prediction["step_id"],
                "actions": _array(root, prediction["actions"]).tolist(),
                "shape": prediction["actions"]["shape"], "dtype": prediction["actions"]["dtype"],
                "horizon": prediction["horizon"], "action_spec": prediction["action_spec"],
                "validity": prediction["validity"], "confidence": prediction["confidence"],
                "raw_output": prediction["raw_output"],
                "executed": [{"step_id": row["step_id"], "selected_chunk_index": row["selected_chunk_index"],
                    "requested": _array(root, row["requested_action"]).tolist(),
                    "applied": _array(root, row["applied_action"]).tolist(),
                    "modification_reason": row["modification_reason"]} for row in actions]},
            "latency_ms": prediction["inference_duration_ns"] / 1e6,
        })
    return {**episode, "records": records, "counts": {
        "observations": len(trace["observations"]), "predictions": len(records),
        "predicted_actions": sum(row["horizon"] for row in trace["policy_predictions"]),
        "executed_actions": len(trace["executed_actions"]),
    }}


def build_eval_dashboard(source, model):
    tasks = [{**task, "episodes": [_eval_episode(source, episode) for episode in task["episodes"]]} for task in model["tasks"]]
    episodes = [episode for task in tasks for episode in task["episodes"]]
    plan = _read(source / "plan.json")
    connection = _read(source / "connection.json") if (source / "connection.json").exists() else {}
    return {
        "schema_version": SCHEMA, "mode": "eval", "run_id": model["run"]["run_id"],
        "status": model["run"]["status"], "tasks": tasks, "outcome": model["outcome"],
        "counts": {"tasks": len(tasks), "episodes": len(episodes), **{
            key: sum(episode["counts"][key] for episode in episodes)
            for key in ("observations", "predictions", "predicted_actions", "executed_actions")}},
        "hyperparameters": {"policy_configuration": plan.get("metadata", {}).get("policy_configuration", {}),
            "policy": connection.get("policy"), "policy_capabilities": connection.get("metadata", {}).get("policy_capabilities"),
            "run_plan": plan},
        "performance": model.get("performance"),
        "notes": ["Eval batch = one canonical policy call with one observation; an action chunk may contain multiple future actions.",
            "Observations include reset and post-action frames, including frames with no policy call.",
            "Raw RGB previews are bounded to the first three predictions per episode. Every record retains image shape, dtype and canonical array reference.",
            "Loss and token accuracy are unavailable unless recorded by the policy; task outcome is a separate measure."],
    }


def build_train_dashboard(source, model):
    plan = _read(source / "training-plan.json")
    scientific = plan.get("scientific", {})
    metrics = [json.loads(line) for line in (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence_path = source / "data-flow.json"
    evidence = _read(evidence_path) if evidence_path.is_file() else None
    training = scientific.get("training", scientific.get("profile", {}).get("training", {}))
    recorded_counts = all(isinstance(row.get("examples_seen"), int) for row in metrics) and bool(metrics)
    return {
        "schema_version": SCHEMA, "mode": "train", "run_id": model["training_run_id"],
        "status": model["status"], "tasks": [], "metrics": metrics, "evidence": evidence,
        "counts": {"optimizer_steps": len(metrics), "examples_seen": sum(row["examples_seen"] for row in metrics) if recorded_counts else None,
            "dataset_records": evidence.get("dataset_records") if evidence else None,
            "supervised_action_tokens": sum(row["action_token_count"] for row in metrics) if metrics and all("action_token_count" in row for row in metrics) else None,
            "microbatches": sum(row["microbatches"] for row in metrics) if metrics and all("microbatches" in row for row in metrics) else None},
        "hyperparameters": {"training": training, "dataset": scientific.get("dataset"),
            "base_checkpoint": scientific.get("base_checkpoint"), "trainer_contract": scientific.get("trainer_contract"),
            "optimizer": evidence.get("optimizer") if evidence else None,
            "profile": scientific.get("profile")},
        "parameter_counts": model["parameter_counts"], "performance": model["system_metrics"],
        "notes": ["Examples seen count includes repeats in the RLDS stream and is separate from the dataset record count.",
            "A microbatch is one forward pass. An optimizer batch accumulates the configured number of microbatches.",
            "Historical runs without data-flow evidence show unavailable input/output samples.",
            "Token accuracy uses supervised next-token targets; it is separate from robot task success. Training runs do not contain robot evaluation success rates."],
    }


def _page(data, prefix, links, source_href):
    payload = _bytes(data).decode("utf-8").replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    navigation = "".join(f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>' for label, href in links)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OVLAB · {html.escape(data["run_id"])} · {data["mode"]}</title><link rel="stylesheet" href="{prefix}assets/dashboard.css"></head>
<body><aside><a class="brand" href="{prefix}index.html">OVLAB<span>Interactive reports</span></a><nav><a href="#overview">Overview</a><a href="#flow">Model data flow</a><a href="#explorer">Batch &amp; record explorer</a><a href="#parameters">Hyperparameters</a><a href="#performance">Performance</a>{navigation}</nav><a href="{html.escape(source_href, quote=True)}">Existing HTML report ↗</a><a href="{prefix}report.json">Dashboard data ↗</a></aside>
<main><header><p class="eyebrow">OVLAB / {data["mode"].upper()} / {html.escape(data.get("scope_label", "Run overview"))}</p><h1>Model evidence dashboard</h1><p class="identity">{html.escape(data["run_id"])}</p><span class="badge">{html.escape(str(data["status"]))}</span></header>
<noscript><p>Enable JavaScript for record selection and charts. Full evidence is available in report.json and the existing HTML report.</p></noscript>
<section id="overview"><h2>Dataset at a glance</h2><div id="kpis" class="grid"></div><div id="notes" class="notes"></div></section>
<section id="flow"><h2>What enters and leaves the model?</h2><div id="flow-content" class="flow"></div></section>
<section id="explorer"><h2>From the full run to a single record</h2><div id="selectors" class="toolbar"></div><div id="batch" class="notes"></div><div id="record" class="grid detail"></div></section>
<section id="parameters"><h2>Hyperparameters · model settings</h2><label>Find a setting <input id="parameter-search" type="search" placeholder="learning rate, rank, precision…"></label><div class="table-scroll"><table><thead><tr><th>Setting / canonical path</th><th>Value</th></tr></thead><tbody id="parameter-rows"></tbody></table></div></section>
<section id="performance"><h2>How did the model perform?</h2><div id="outcome"></div><div id="charts" class="grid"></div><div id="metric-table" class="table-scroll"></div></section>
<footer>Offline · canonical evidence stays read-only · missing values remain unavailable · <a href="{prefix}manifest.json">Integrity manifest</a></footer></main>
<script id="dashboard-data" type="application/json">{payload}</script><script src="{prefix}assets/dashboard.js" defer></script></body></html>'''


def publish_dashboard(report: Path, derived_root: Path, source: Path, *, training=False):
    target = dashboard_path(report, derived_root)
    if target.exists():
        verify_dashboard(report, derived_root)
        return target
    stage = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    (stage / "assets").mkdir(parents=True)
    try:
        model = _read(report / "report.json")
        data = build_train_dashboard(source, model) if training else build_eval_dashboard(source, model)
        for name, content in _assets().items():
            (stage / "assets" / name).write_bytes(content)
        (stage / "report.json").write_bytes(_bytes(data))
        links = []
        for task in data["tasks"]:
            task_dir = stage / "tasks" / task["task_key"]
            task_dir.mkdir(parents=True)
            scoped = {**data, "tasks": [task], "outcome": task["outcome"], "scope_label": task["task_id"],
                "counts": {"tasks": 1, "episodes": len(task["episodes"]), **{
                    key: sum(episode["counts"][key] for episode in task["episodes"])
                    for key in ("observations", "predictions", "predicted_actions", "executed_actions")}}}
            (task_dir / "index.html").write_text(_page(scoped, "../../", [("Run overview", "../../index.html")],
                os.path.relpath(report / "tasks" / task["task_key"] / "index.html", task_dir)), encoding="utf-8")
            links.append((task["task_id"], f'tasks/{task["task_key"]}/index.html'))
        # The run view contains bounded previews; task views retain every prediction.
        overview = {**data, "tasks": [{**task, "episodes": [{**episode, "records": episode["records"][:3]}
            for episode in task["episodes"]]} for task in data["tasks"]]}
        (stage / "index.html").write_text(_page(overview, "", links, os.path.relpath(report / "index.html", stage)), encoding="utf-8")
        files = [{"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(stage.rglob("*")) if path.is_file()]
        manifest = {"schema_version": SCHEMA, "derived_build_id": target.name, "identity_inputs": _identity(report), "files": files}
        manifest["manifest_payload_sha256"] = hashlib.sha256(_bytes(manifest)).hexdigest()
        (stage / "manifest.json").write_bytes(_bytes(manifest))
        stage.rename(target)
        temporary = target.parent / f".latest.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(_bytes({"schema_version": SCHEMA, "derived_build_id": target.name}))
        temporary.replace(target.parent / "latest.json")
        # Only the immutable build itself is finalized. Its parents, including
        # the host-owned exports/interactive bind mount, stay writable so later
        # builds can publish alongside it and update latest.json.
        finalize_managed_tree(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_dashboard(report, derived_root)
    return target


def verify_dashboard(report: Path, derived_root: Path):
    target = dashboard_path(report, derived_root)
    manifest = _read(target / "manifest.json")
    digest = manifest.pop("manifest_payload_sha256", None)
    if digest != hashlib.sha256(_bytes(manifest)).hexdigest() or manifest.get("identity_inputs") != _identity(report):
        raise ArtifactError("interactive report manifest checksum or identity mismatch")
    if manifest.get("schema_version") != SCHEMA or manifest.get("derived_build_id") != target.name:
        raise ArtifactError("interactive report build identity mismatch")
    expected = {"manifest.json"}
    for row in manifest["files"]:
        path = (target / row["path"]).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file() or path.stat().st_size != row["size_bytes"] or _sha(path) != row["sha256"]:
            raise ArtifactError(f"interactive report file checksum mismatch: {row['path']}")
        expected.add(row["path"])
    if expected != {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}:
        raise ArtifactError("interactive report contains missing or unexpected files")
    return target
