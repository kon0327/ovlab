"""Plan and execute a tree of independently runnable experiment nodes.

The JSON or YAML document is an orchestration layer over the existing, validated
training profiles and evaluation experiments.  Every selected phase resolves
to an ordinary OVLAB YAML configuration before any workload is started.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from urllib.parse import quote

from .config_bundle import ConfigBundleBuilder
from .errors import ConfigError, StrictYamlError
from .strict_yaml import dumps, loads as load_yaml
from .training_composition import load_training_profile
from .training_profiles import TrainingProfile
from .schema import validate
from ovlab_core.experiment_scope import ExperimentScope


SCHEMA = "ovlab.experiment/v1"
PLAN_SCHEMA = "ovlab.experiment-plan/v1"
RESULT_SCHEMA = "ovlab.experiment-result/v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PHASES = ("train", "evaluate")
_DEFAULT_WORKERS = 2
_MAX_WORKERS = 32
_RESERVED = {"train", "evaluate", "derived", "interactive", "isolated", "grouped"}
_COMPONENTS = {
    "benchmark": "benchmark", "policy": "policy", "metrics": "metric_set",
    "protocol": "protocol", "artifacts": "artifact_store",
}
_COMPONENT_DIRS = {
    "benchmark": "benchmarks", "policy": "policies", "metrics": "metrics",
    "protocol": "protocols", "artifacts": "artifacts",
}


class ExperimentSpecError(ConfigError):
    """An experiment tree cannot be planned safely."""


class ExperimentExecutionError(RuntimeError):
    """A planned experiment phase or required report failed."""


class _StepFailure(Exception):
    def __init__(self, cause: BaseException, record: dict[str, object] | None):
        super().__init__(str(cause))
        self.cause = cause
        self.record = record


def _object_pairs(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ExperimentSpecError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _keys(value, allowed, required, label):
    if not isinstance(value, dict):
        raise ExperimentSpecError(f"{label} must be an object")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ExperimentSpecError(f"{label}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _safe_id(value, label):
    if not isinstance(value, str) or len(value) > 120 or _SAFE.fullmatch(value) is None or value in _RESERVED:
        raise ExperimentSpecError(f"{label} must be a portable identifier")
    return value


def _merge(base, patch):
    """Object fields merge recursively; arrays and scalars replace entirely."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return deepcopy(patch)
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = _merge(result[key], value) if key in result else deepcopy(value)
    return result


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix


def _field_sources(base, reference, patches):
    sources = {field: reference for field in _leaves(base)}
    for node_path, patch in patches:
        for field in _leaves(patch):
            for prior in tuple(sources):
                if prior == field or prior.startswith(field + ".") or field.startswith(prior + "."):
                    del sources[prior]
            sources[field] = f"experiment:{node_path or '<root>'}:{field}"
    return sources


def _read_spec_source(path: Path):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentSpecError(f"cannot read experiment spec: {path}") from exc
    if len(raw) > 2_000_000:
        raise ExperimentSpecError("experiment spec exceeds 2 MB")
    is_yaml = path.suffix.lower() in {".yaml", ".yml"}
    try:
        if is_yaml:
            document = load_yaml(raw.decode("utf-8"), source=str(path))
        else:
            document = json.loads(raw, object_pairs_hook=_object_pairs)
    except (UnicodeError, json.JSONDecodeError, StrictYamlError) as exc:
        raise ExperimentSpecError(f"invalid experiment {'YAML' if is_yaml else 'JSON'}: {exc}") from exc
    return document, raw


def _read_spec(path: Path):
    return _read_spec_source(path)[0]


def _spec_hash(document):
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_config(config, label):
    _keys(config, _PHASES, (), label)
    for phase, value in config.items():
        allowed = ("profile", "patch", "component_patches") if phase == "evaluate" else ("profile", "patch")
        _keys(value, allowed, (), f"{label}.{phase}")
        if "profile" in value and (not isinstance(value["profile"], str) or not value["profile"]):
            raise ExperimentSpecError(f"{label}.{phase}.profile must be a nonempty path/name")
        if "patch" in value and not isinstance(value["patch"], dict):
            raise ExperimentSpecError(f"{label}.{phase}.patch must be an object")
        if "component_patches" in value:
            _keys(value["component_patches"], _COMPONENTS, (), f"{label}.{phase}.component_patches")
            if any(not isinstance(patch, dict) for patch in value["component_patches"].values()):
                raise ExperimentSpecError(f"{label}.{phase}.component_patches values must be objects")


def _validate_node(raw, path="", depth=0, counter=None):
    if counter is None:
        counter = [0]
    counter[0] += 1
    if depth > 12 or counter[0] > 512:
        raise ExperimentSpecError("experiment tree exceeds 12 levels or 512 nodes")
    allowed = {"id", "name", "config", "execute", "depends_on", "children"}
    if depth == 0:
        allowed.update(("schema_version", "kind"))
    _keys(raw, allowed, ("id",) if depth else ("schema_version", "kind", "id", "name"), path or "root")
    if depth == 0 and raw["schema_version"] != SCHEMA:
        raise ExperimentSpecError(f"schema_version must be {SCHEMA}")
    if depth == 0 and raw["kind"] != "workflow":
        raise ExperimentSpecError("kind must be workflow")
    _safe_id(raw["id"], f"{path or 'root'}.id")
    if "name" in raw and (not isinstance(raw["name"], str) or not raw["name"].strip()):
        raise ExperimentSpecError(f"{path or 'root'}.name must be a nonempty string")
    if "config" in raw:
        _validate_config(raw["config"], f"{path or 'root'}.config")
    execute = raw.get("execute", [])
    if not isinstance(execute, list) or any(item not in _PHASES for item in execute) or len(execute) != len(set(execute)):
        raise ExperimentSpecError(f"{path or 'root'}.execute must list unique train/evaluate phases")
    dependencies = raw.get("depends_on", [])
    if not isinstance(dependencies, list):
        raise ExperimentSpecError(f"{path or 'root'}.depends_on must be an array")
    for index, dependency in enumerate(dependencies):
        _keys(dependency, ("node", "phase"), ("node", "phase"), f"{path}.depends_on[{index}]")
        node = dependency["node"]
        if not isinstance(node, str) or any(_SAFE.fullmatch(part) is None for part in node.split("/") if part) or "//" in node or node.startswith("/") or node.endswith("/"):
            raise ExperimentSpecError(f"{path}.depends_on[{index}].node is invalid")
        if dependency["phase"] not in _PHASES:
            raise ExperimentSpecError(f"{path}.depends_on[{index}].phase is invalid")
    children = raw.get("children", [])
    if not isinstance(children, list):
        raise ExperimentSpecError(f"{path or 'root'}.children must be an array")
    names = set()
    for child in children:
        if not isinstance(child, dict) or "id" not in child:
            raise ExperimentSpecError(f"{path or 'root'}.children entries need an id")
        child_id = _safe_id(child["id"], "child.id")
        if child_id in names:
            raise ExperimentSpecError(f"duplicate child id {child_id!r} below {path or '<root>'}")
        names.add(child_id)
        _validate_node(child, f"{path}/{child_id}".strip("/"), depth + 1, counter)


def _apply_phase_config(inherited, node_config, node_path):
    state = deepcopy(inherited)
    for phase, value in node_config.items():
        current = state.setdefault(phase, {"profile": None, "patches": [], "component_patches": {}})
        if "profile" in value:
            current["profile"] = value["profile"]
            current["profile_source"] = node_path or "<root>"
        if value.get("patch"):
            current["patches"].append((node_path, value["patch"]))
        for kind, patch in value.get("component_patches", {}).items():
            current["component_patches"].setdefault(kind, []).append((node_path, patch))
    return state


def _visit(raw, inherited=None, path=""):
    state = _apply_phase_config(inherited or {}, raw.get("config", {}), path)
    yield raw, path, state
    for child in raw.get("children", []):
        child_path = f"{path}/{child['id']}".strip("/")
        yield from _visit(child, state, child_path)


def _dependency_graph(nodes, selected):
    tasks = {(node["path"], phase): (node, phase) for node in nodes for phase in node["execute"] if node["path"] in selected}
    edges = {key: set() for key in tasks}
    for (path, phase), (node, _) in tasks.items():
        if phase == "evaluate" and (path, "train") in tasks:
            edges[(path, phase)].add((path, "train"))
        for dependency in node["depends_on"]:
            key = dependency["node"], dependency["phase"]
            if key not in tasks:
                raise ExperimentSpecError(f"{path or '<root>'} depends on unselected/missing phase {key}")
            if key == (path, phase):
                raise ExperimentSpecError("a phase cannot depend on itself")
            edges[(path, phase)].add(key)
    return tasks, edges


def _topological_steps(nodes, selected):
    tasks, edges = _dependency_graph(nodes, selected)
    result = []
    pending = {key: set(value) for key, value in edges.items()}
    while pending:
        ready = next((key for key, dependencies in pending.items() if not dependencies), None)
        if ready is None:
            raise ExperimentSpecError("experiment phase dependency cycle")
        node, phase = tasks[ready]
        result.append({"node": node["path"], "phase": phase})
        del pending[ready]
        for dependencies in pending.values():
            dependencies.discard(ready)
    return result


def _write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.chmod(0o640)
    os.replace(temporary, path)


def _export_link(export_root: Path, candidate: str | Path | None):
    """Use only existing files within this experiment's export directory."""
    if not candidate:
        return None
    target = Path(candidate).expanduser().resolve()
    root = export_root.resolve()
    if not target.is_relative_to(root):
        return None
    if target.is_dir():
        target = next((target / name for name in (
            "index.html", "summary.json", "comparison.json", "metadata.json",
        ) if (target / name).is_file()), target)
    if not target.is_file():
        return None
    return "/".join(quote(part, safe="") for part in target.relative_to(root).parts)


def _latest_report(export_root: Path, node_root: Path, run_id: str, profile: str):
    profile_root = node_root / "interactive" / run_id / profile
    latest = profile_root / "latest.json"
    if not latest.is_file():
        return None
    try:
        build = json.loads(latest.read_text(encoding="utf-8")).get("build_id")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(build, str) or _SAFE.fullmatch(build) is None:
        return None
    target = profile_root / build / "index.html"
    return target if _export_link(export_root, target) else None


def _comparison_status(grouped, phase: str, export_root: Path):
    output = grouped.get("result", {}).get("output")
    if not isinstance(output, str):
        return "Unavailable"
    path = Path(output).expanduser().resolve()
    if not path.is_relative_to(export_root.resolve()):
        return "Unavailable"
    source = path / ("comparison.json" if phase == "train" else "metadata.json")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Recorded; comparison details unavailable"
    comparison = document if phase == "train" else document.get("comparison", {})
    status = comparison.get("status") if isinstance(comparison, dict) else None
    return str(status).replace("_", " ") if status else "Recorded; comparison status unavailable"


def build_experiment_index(plan, result, export_root: Path):
    """Render a static, fully visible experiment tree with safe local links."""
    root = Path(export_root).resolve()
    nodes = {node["path"]: node for node in plan["nodes"]}
    selected = set(plan["selected_nodes"])
    phase_results = result.get("nodes", {})
    grouped = result.get("grouped", {})

    def esc(value):
        return html.escape(str(value), quote=True)

    def link(candidate, label):
        href = _export_link(root, candidate)
        return (f'<a href="{esc(href)}">{esc(label)}</a>' if href
                else f'<span class="unavailable">{esc(label)} unavailable</span>')

    def render_node(path):
        node = nodes[path]
        node_results = phase_results.get(path, {})
        node_export = root
        segments = path.split("/") if path else []
        anchor = "node-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        rows = []
        for phase in _PHASES:
            phase_result = node_results.get(phase)
            if phase_result:
                run_id = phase_result["run_id"]
                node_root = node_export / phase / Path(*segments)
                configured = node.get("configuration", {}).get(phase, {})
                profile = ("system-performance" if phase == "train" else
                           configured.get("document", {}).get("reporting", {}).get("profile", "libero-task-default"))
                report_result = (phase_result.get("result", {}).get("performance_report") if phase == "train" else
                                 phase_result.get("postprocessing", {}).get("report")) or {}
                report_path = report_result.get("interactive_output") if isinstance(report_result, dict) else None
                if not _export_link(root, report_path):
                    report_path = _latest_report(root, node_root, run_id, profile)
                isolated_result = (phase_result.get("isolated_export") if phase == "train" else
                                   phase_result.get("postprocessing", {}).get("isolated_export")) or {}
                isolated_path = isolated_result.get("output") if isinstance(isolated_result, dict) else None
                if not _export_link(root, isolated_path):
                    isolated_path = node_root / "isolated" / run_id
                rows.append(
                    f'<article class="phase"><h3>{esc(phase.title())}</h3>'
                    f'<p><strong>Completed</strong> · Run <code>{esc(run_id)}</code></p>'
                    f'<p>{link(report_path, "Interactive report")} · {link(isolated_path, "Isolated export")}</p></article>'
                )
            else:
                status = ("Pending" if result["status"] == "running" else "Unavailable") if (
                    phase in node["execute"] and path in selected
                ) else ("Not selected" if phase in node["execute"] else "Not scheduled")
                rows.append(f'<article class="phase"><h3>{esc(phase.title())}</h3><p>{esc(status)}</p></article>')
        comparison_rows = []
        for phase in _PHASES:
            item = grouped.get(phase, {}).get(path)
            if item:
                output = item.get("result", {}).get("output")
                comparison_rows.append(
                    f'<p>{esc(phase.title())} comparison: {esc(_comparison_status(item, phase, root))} · '
                    f'{link(output, "Grouped export")}</p>'
                )
            elif node["children"]:
                comparison_rows.append(f'<p>{esc(phase.title())} comparison: Not generated</p>')
        children = "".join(f'<li>{render_node(f"{path}/{child}".strip("/"))}</li>' for child in node["children"])
        child_list = f'<ol class="children">{children}</ol>' if children else ""
        label = path or "Root"
        return (
            f'<section class="node" id="{anchor}"><h2>{esc(node["name"])} '
            f'<span class="path">{esc(label)}</span></h2>'
            f'<div class="phases">{"".join(rows)}</div>'
            f'<div class="comparisons">{"".join(comparison_rows)}</div>{child_list}</section>'
        )

    status = result.get("status", "unknown")
    failure = result.get("failure")
    failure_html = (f'<p class="failure">{esc(failure.get("type", "Failure"))}: '
                    f'{esc(failure.get("message", ""))}</p>') if isinstance(failure, dict) else ""
    navigation = " ".join(
        f'<a href="#node-{hashlib.sha256(node["path"].encode("utf-8")).hexdigest()[:12]}">{esc(node["path"] or "Root")}</a>'
        for node in plan["nodes"]
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(plan["name"])} · OVLAB experiment</title>'
        '<style>body{font:16px/1.5 system-ui,sans-serif;color:#172438;background:#f4f7fb;margin:0}'
        'main{max-width:1200px;margin:auto;padding:2rem 1rem 5rem}a{color:#1255a2}a:focus-visible{outline:3px solid #f59d2a}'
        'h1{font-size:clamp(1.8rem,4vw,2.7rem);margin:.2rem 0}.summary{background:#e7f0fb;padding:1rem;border-radius:12px}'
        '.tree-nav{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.5rem 0}.tree-nav a{background:white;padding:.3rem .7rem;border-radius:8px}'
        '.node{background:white;border:1px solid #c8d7e8;border-radius:12px;padding:1rem;margin:1rem 0}'
        '.node h2{font-size:1.25rem;margin:0 0 .7rem}.path{font:normal .85rem ui-monospace,monospace;color:#52667e;margin-left:.5rem}'
        '.phases{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.7rem}'
        '.phase{background:#f5f8fc;border:1px solid #dde6f0;border-radius:8px;padding:.65rem}.phase h3{margin:0;font-size:1rem}'
        '.phase p,.comparisons p{margin:.3rem 0}.unavailable{color:#63758b}.comparisons{margin-top:.6rem}'
        '.children{border-left:2px solid #ccd9e8;margin:1rem 0 0;padding-left:1.6rem}.children li{padding-left:.2rem}'
        '.failure{background:#fff0ed;border-left:4px solid #b3321b;padding:.7rem}code{overflow-wrap:anywhere}'
        '</style></head><body><main>'
        f'<header><p>OVLAB experiment</p><h1>{esc(plan["name"])}</h1>'
        f'<p><code>{esc(result["experiment_id"])}</code></p></header>'
        f'<div class="summary"><strong>Status: {esc(status)}</strong> · '
        f'{len(plan["nodes"])} nodes · {len(plan["steps"])} planned phases · '
        '<a href="experiment.json">Complete result JSON</a></div>'
        f'{failure_html}<nav class="tree-nav" aria-label="Experiment nodes">{navigation}</nav>'
        f'<div class="tree">{render_node("")}</div>'
        '</main></body></html>\n'
    )


def _publish_experiment_index(plan, result, export_root: Path):
    _write_json(export_root / "experiment.json", result)
    index = export_root / "index.html"
    temporary = index.with_name(f".index.{uuid.uuid4().hex}.tmp")
    temporary.write_text(build_experiment_index(plan, result, export_root), encoding="utf-8")
    temporary.chmod(0o640)
    os.replace(temporary, index)


class ExperimentWorkflow:
    def __init__(self, repository_root: str | Path, *, environment=None, runner=None):
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.config_root = self.repository_root / "configs"
        self.environment = dict(os.environ if environment is None else environment)
        self.runner = runner
        self.bundle = ConfigBundleBuilder(self.repository_root)
        self._process_lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()
        self._cancelled = threading.Event()

    def _spec_path(self, reference: str | Path) -> Path:
        path = Path(reference).expanduser()
        if path.is_absolute():
            return path.resolve()
        local = path.resolve()
        return local if local.is_file() else (self.repository_root / path).resolve()

    def _profile_path(self, reference, phase):
        path = Path(reference).expanduser()
        if path.is_absolute():
            candidate = path.resolve()
        elif path.parts and path.parts[0] in {"configs", "experiments"}:
            candidate = (self.repository_root / path).resolve()
        else:
            candidate = (self.repository_root / "experiments" / ("training" if phase == "train" else "evaluation") / path).resolve()
            if not candidate.is_file() and not candidate.suffix:
                candidate = candidate.with_suffix(".yaml")
        section = self.repository_root / "experiments" / ("training" if phase == "train" else "evaluation")
        if not candidate.is_relative_to(section.resolve()) or not candidate.is_file() or candidate.is_symlink():
            raise ExperimentSpecError(f"{phase} profile must be a regular file inside {section}: {reference}")
        return candidate

    def _resolved_config(self, phase, state):
        path = self._profile_path(state["profile"], phase)
        reference = path.relative_to(self.repository_root).as_posix()
        if phase == "train":
            base = dict(load_training_profile(path, config_root=self.repository_root / "experiments").document)
        else:
            base = self.bundle.load_component(reference, "experiment")
        document = deepcopy(base)
        for _, patch in state["patches"]:
            document = _merge(document, patch)
        if phase == "train":
            document = dict(TrainingProfile.from_document(document).document)
        else:
            if "extends" in document:
                document.pop("extends")
            validate(document, reference, "experiment")
        result = {
            "profile": reference,
            "profile_source": state.get("profile_source", "<root>"),
            "document": document,
            "field_sources": _field_sources(base, reference, state["patches"]),
        }
        if phase == "evaluate":
            component_documents = {}
            for kind, patches in state["component_patches"].items():
                component_ref = document["components"][kind]
                component_base = self.bundle.load_component(component_ref, _COMPONENTS[kind])
                component = deepcopy(component_base)
                for _, patch in patches:
                    component = _merge(component, patch)
                component.pop("extends", None)
                validate(component, component_ref, _COMPONENTS[kind])
                component_documents[kind] = {
                    "profile": component_ref,
                    "document": component,
                    "field_sources": _field_sources(component_base, component_ref, patches),
                }
            result["component_documents"] = component_documents
            # Verify every unpatched reference without requiring runtime resources.
            for kind, component_ref in document["components"].items():
                if kind not in component_documents:
                    self.bundle.load_component(component_ref, _COMPONENTS[kind])
            self.bundle.load_component(document["resources"]["registry"], "resource_registry")
        return result

    def plan(self, spec: str | Path, *, node: str | None = None, node_only=False,
             sequential=False, max_workers=None):
        if sequential and max_workers is not None:
            raise ExperimentSpecError("--sequential and --max-workers cannot be combined")
        workers = 1 if sequential else (_DEFAULT_WORKERS if max_workers is None else max_workers)
        if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
            raise ExperimentSpecError(f"max workers must be between 1 and {_MAX_WORKERS}")
        spec_path = self._spec_path(spec)
        raw = _read_spec(spec_path)
        _validate_node(raw)
        if node is not None and node and any(_SAFE.fullmatch(part) is None for part in node.split("/")):
            raise ExperimentSpecError("--node must be a slash-separated path of node IDs")
        root_id = _safe_id(raw["id"], "root.id")
        spec_hash = _spec_hash(raw)
        entries = list(_visit(raw))
        paths = {path for _, path, _ in entries}
        selected_path = node or ""
        if selected_path not in paths:
            raise ExperimentSpecError(f"unknown experiment node: {selected_path}")
        selected = {selected_path} if node_only else {
            path for path in paths
            if not selected_path or path == selected_path or path.startswith(selected_path + "/")
        }
        nodes = []
        for item, path, state in entries:
            execute = item.get("execute", [])
            configuration = {}
            if path in selected:
                for phase in _PHASES:
                    phase_state = state.get(phase)
                    if phase_state and phase_state.get("profile"):
                        configuration[phase] = self._resolved_config(phase, phase_state)
                    elif phase in execute:
                        raise ExperimentSpecError(f"{path or '<root>'} executes {phase} without an inherited profile")
            if path in selected and "evaluate" in execute and not configuration["evaluate"]["document"].get("reporting", {}).get("enabled", True):
                raise ExperimentSpecError(
                    f"{path or '<root>'} disables evaluation reporting; set config.evaluate.patch.reporting.enabled to true"
                )
            nodes.append({
                "id": item["id"], "name": item.get("name", item["id"]), "path": path,
                "parent": path.rpartition("/")[0] if path else None,
                "execute": list(execute), "depends_on": deepcopy(item.get("depends_on", [])),
                "configuration": configuration,
                "children": [child["id"] for child in item.get("children", [])],
            })
        steps = _topological_steps(nodes, selected)
        return {
            "schema_version": PLAN_SCHEMA, "spec_schema_version": SCHEMA,
            "spec": str(spec_path), "spec_hash": spec_hash, "id": root_id,
            "name": raw["name"], "selected_node": selected_path,
            "node_only": bool(node_only), "nodes": nodes, "steps": steps,
            "selected_nodes": sorted(selected),
            "execution": {"mode": "sequential" if workers == 1 else "parallel", "max_workers": workers},
            "dependency_semantics": "scheduling only; checkpoints are not wired between phases automatically",
            "side_effects_performed": False,
        }

    def _materialize(self, instance, node_path, phase, configuration):
        # Evaluation deployment accepts only experiments/evaluation YAML;
        # temporary configurations are removed after this run. All bytes are
        # separately snapshotted in the immutable experiment plan.
        parts = ["nodes", *node_path.split("/")] if node_path else ["root"]
        base = self.repository_root / "experiments" / ("training" if phase == "train" else "evaluation") / ".generated" / instance
        target = base.joinpath(*parts[:-1], parts[-1] + ".yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        document = deepcopy(configuration["document"])
        if phase == "evaluate":
            for kind, component in configuration.get("component_documents", {}).items():
                component_root = self.config_root / _COMPONENT_DIRS[kind] / ".generated" / instance
                component_path = component_root.joinpath(*parts[:-1], parts[-1] + ".yaml")
                component_path.parent.mkdir(parents=True, exist_ok=True)
                component_path.write_text(dumps(component["document"]), encoding="utf-8")
                document["components"][kind] = component_path.relative_to(self.config_root).as_posix()
        target.write_text(dumps(document), encoding="utf-8")
        return target

    def _invoke(self, command, env):
        invocation = [str(self.repository_root / "ovlab"), *command, "--json"]
        if self.runner is not None:
            completed = self.runner(invocation, cwd=self.repository_root, env=env,
                                    stdout=subprocess.PIPE, stderr=sys.stderr, text=True, check=False)
        else:
            with self._process_lock:
                if self._cancelled.is_set():
                    raise ExperimentExecutionError("experiment execution was cancelled")
                process = subprocess.Popen(invocation, cwd=self.repository_root, env=env,
                                           stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                                           start_new_session=True)
                self._active_processes.add(process)
            try:
                stdout, _ = process.communicate()
                completed = subprocess.CompletedProcess(invocation, process.returncode, stdout)
            finally:
                with self._process_lock:
                    self._active_processes.discard(process)
        try:
            output = json.loads(completed.stdout.strip())
        except (AttributeError, json.JSONDecodeError) as exc:
            raise ExperimentExecutionError(f"{command[0]} command returned no parseable JSON result") from exc
        if completed.returncode != 0 or output.get("status") != "success":
            errors = output.get("errors", [])
            message = "; ".join(str(item.get("message", item)) for item in errors) or f"exit {completed.returncode}"
            raise ExperimentExecutionError(f"{' '.join(command[:2])} failed: {message}")
        return output["result"]

    def _cancel_running(self):
        self._cancelled.set()
        with self._process_lock:
            for process in tuple(self._active_processes):
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass

    def _execute_step(self, instance, node_path, phase, item, runs_root, exports_root):
        record = None
        try:
            configuration = item["configuration"][phase]
            generated = self._materialize(instance, node_path, phase, configuration)
            env = dict(
                self.environment, OVLAB_RUNS_ROOT=str(runs_root), OVLAB_EXPORTS_ROOT=str(exports_root),
                OVLAB_EXPERIMENT_ID=instance, OVLAB_EXPERIMENT_NODE=node_path,
            )
            if phase == "train":
                command = ["train", "run", str(generated)]
            elif "deployment" in configuration["document"]:
                command = ["eval", "run", str(generated)]
            else:
                command = ["run", str(generated)]
            child = self._invoke(command, env)
            run_path = child.get("host_path") or child.get("run_path")
            if run_path is None and child.get("new_run_paths"):
                run_path = child["new_run_paths"][0]
            if not isinstance(run_path, str):
                raise ExperimentExecutionError(f"{node_path or '<root>'} {phase} did not return a run path")
            scope = ExperimentScope(instance, tuple(node_path.split("/")) if node_path else ())
            expected_parent = scope.runs_root(runs_root, phase).resolve()
            if Path(run_path).resolve().parent != expected_parent:
                raise ExperimentExecutionError(
                    f"{node_path or '<root>'} {phase} returned a run outside {expected_parent}: {run_path}"
                )
            run_id = Path(run_path).name
            record = {"run_id": run_id, "run_path": run_path, "result": child}
            if phase == "train":
                record["isolated_export"] = self._invoke([
                    "export", "isolated", "--category", "train", "--run", run_id,
                ], env)
            elif command[0] == "run":
                profile = configuration["document"].get("reporting", {}).get("profile", "libero-task-default")
                record["postprocessing"] = self._invoke([
                    "report", "publish", "--run", run_id, "--profile", profile,
                ], env)
            return record
        except BaseException as exc:
            raise _StepFailure(exc, record) from exc

    def _execute_steps(self, plan, result, manifest, export_root, runs_root, exports_root):
        nodes = {item["path"]: item for item in plan["nodes"]}
        _, dependencies = _dependency_graph(plan["nodes"], set(plan["selected_nodes"]))
        order = [(step["node"], step["phase"]) for step in plan["steps"]]
        order_index = {key: index for index, key in enumerate(order)}
        pending, completed, active = set(order), set(), {}
        first_error = None
        workers = plan["execution"]["max_workers"]

        def save(key, record):
            if record is not None:
                node_path, phase = key
                result["nodes"].setdefault(node_path, {})[phase] = record
            _write_json(manifest / "result.json", result)
            _publish_experiment_index(plan, result, export_root)

        def collect(future, key):
            nonlocal first_error
            try:
                record = future.result()
            except _StepFailure as exc:
                result.setdefault("failures", []).append({
                    "node": key[0], "phase": key[1],
                    "type": type(exc.cause).__name__, "message": str(exc.cause),
                })
                save(key, exc.record)
                if first_error is None:
                    first_error = exc.cause
                    self._cancel_running()
            else:
                save(key, record)
                completed.add(key)

        self._cancelled.clear()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ovlab-experiment") as executor:
            try:
                while pending or active:
                    if first_error is None:
                        for key in order:
                            if len(active) >= workers:
                                break
                            if key in pending and dependencies[key].issubset(completed):
                                pending.remove(key)
                                node_path, phase = key
                                future = executor.submit(self._execute_step, result["experiment_id"],
                                                         node_path, phase, nodes[node_path], runs_root, exports_root)
                                active[future] = key
                    if not active:
                        if first_error is not None:
                            raise first_error
                        raise ExperimentExecutionError("experiment scheduler has no runnable phase")
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in sorted(finished, key=lambda item: order_index[active[item]]):
                        key = active.pop(future)
                        collect(future, key)
                if first_error is not None:
                    raise first_error
            except BaseException:
                self._cancel_running()
                for future in active:
                    future.cancel()
                for future in tuple(active):
                    if future.cancelled():
                        continue
                    key = active[future]
                    try:
                        collect(future, key)
                    except BaseException:
                        pass
                raise

    def _grouped(self, plan, result, env):
        """Publish comparison reports for each node with multiple child runs."""
        outputs = {phase: {} for phase in _PHASES}
        for phase in _PHASES:
            completed = {path: item[phase]["run_id"] for path, item in result["nodes"].items()
                         if phase in item and item[phase].get("run_id")}
            for node in reversed(plan["nodes"]):
                path = node["path"]
                child_runs = []
                represented_children = 0
                for child in node["children"]:
                    child_path = f"{path}/{child}".strip("/")
                    branch = [run_id for run_path, run_id in completed.items()
                              if run_path == child_path or run_path.startswith(child_path + "/")]
                    represented_children += bool(branch)
                    child_runs.extend(branch)
                child_runs = list(dict.fromkeys(child_runs))
                if represented_children < 2 or len(child_runs) < 2:
                    continue
                group_env = dict(env, OVLAB_EXPERIMENT_NODE=path)
                name = (node["id"] if path else plan["id"]) + "-comparison"
                grouped = self._invoke([
                    "export", "grouped", "--category", phase, "--name", name, "--runs", *child_runs,
                ], group_env)
                outputs[phase][path] = {"status": "completed", "run_ids": child_runs, "result": grouped}
        return outputs

    def run(self, spec: str | Path, *, node: str | None = None, node_only=False,
            dry_run=False, sequential=False, max_workers=None):
        plan = self.plan(spec, node=node, node_only=node_only, sequential=sequential, max_workers=max_workers)
        if dry_run:
            return plan
        if not plan["steps"]:
            raise ExperimentSpecError("selected experiment contains no executable train/evaluate phases")
        spec_path = Path(plan["spec"])
        spec_snapshot, source_bytes = _read_spec_source(spec_path)
        if _spec_hash(spec_snapshot) != plan["spec_hash"]:
            raise ExperimentSpecError("experiment specification changed during planning; retry the command")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        instance = f"{plan['id']}_{stamp}_{plan['spec_hash'][:6]}{uuid.uuid4().hex[:4]}"
        data_root = Path(self.environment.get("OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data")).expanduser().resolve()
        runs_root = Path(self.environment.get("OVLAB_RUNS_ROOT", data_root / "runs")).expanduser().resolve()
        exports_root = Path(self.environment.get("OVLAB_EXPORTS_ROOT", runs_root.parent / "exports")).expanduser().resolve()
        for global_root in (runs_root, exports_root):
            global_root.mkdir(parents=True, exist_ok=True)
            if global_root.stat().st_uid == os.getuid():
                global_root.chmod(0o2770)
        experiment_root = runs_root / instance
        export_root = exports_root / instance
        experiment_root.mkdir(parents=True, exist_ok=False)
        export_root.mkdir(parents=True, exist_ok=False)
        for root in (experiment_root, export_root):
            root.chmod(0o2770)
            for phase in _PHASES:
                child = root / phase
                child.mkdir(parents=True, exist_ok=True)
                child.chmod(0o2770)
        manifest = experiment_root / ".experiment"
        manifest.mkdir()
        manifest.chmod(0o2770)
        _write_json(manifest / "spec.json", spec_snapshot)
        if spec_path.suffix.lower() in {".yaml", ".yml"}:
            original = manifest / "spec.yaml"
            original.write_bytes(source_bytes)
            original.chmod(0o640)
        _write_json(manifest / "plan.json", {**plan, "experiment_id": instance})
        result = {
            "schema_version": RESULT_SCHEMA, "experiment_id": instance,
            "spec_hash": plan["spec_hash"], "status": "running", "nodes": {},
            "grouped": {}, "runs_root": str(experiment_root), "exports_root": str(export_root),
            "overview": str(export_root / "index.html"),
            "execution": plan["execution"],
        }
        _write_json(manifest / "result.json", result)
        _publish_experiment_index(plan, result, export_root)
        try:
            self._execute_steps(plan, result, manifest, export_root, runs_root, exports_root)
            result["grouped"] = self._grouped(plan, result, dict(
                self.environment, OVLAB_RUNS_ROOT=str(runs_root), OVLAB_EXPORTS_ROOT=str(exports_root),
                OVLAB_EXPERIMENT_ID=instance,
            ))
            result["status"] = "completed"
            _write_json(manifest / "result.json", result)
            _publish_experiment_index(plan, result, export_root)
            return result
        except BaseException as exc:
            result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(manifest / "result.json", result)
            _publish_experiment_index(plan, result, export_root)
            raise
        finally:
            # Profiles and patched components are temporary execution inputs;
            # their exact resolved documents remain in plan.json and each run.
            for section in ("experiments/training", "experiments/evaluation", "benchmarks", "policies", "metrics", "protocols", "artifacts"):
                generated_root = (self.repository_root if section.startswith("experiments/") else self.config_root) / section / ".generated" / instance
                if generated_root.is_dir():
                    shutil.rmtree(generated_root)
                    try:
                        generated_root.parent.rmdir()
                    except OSError:
                        pass
