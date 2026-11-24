"""Resolve training, evaluation and workflow sources into one experiment tree.

The JSON or YAML document is an orchestration layer over the existing, validated
training profiles and evaluation experiments.  Every selected phase resolves
to an ordinary OVLAB YAML configuration before any workload is started.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
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

from .config_bundle import ConfigBundleBuilder
from .errors import ConfigError, StrictYamlError
from .strict_yaml import dumps, loads as load_yaml
from .training_composition import load_training_document, load_training_profile
from .training_profiles import TrainingProfile
from .schema import validate
from .policy_plugins import with_registry
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


class ExperimentIntegrityError(RuntimeError):
    """An experiment's stored metadata or canonical evidence is inconsistent."""


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
        allowed = ("profile", "patch", "component_patches", "checkpoint_from") if phase == "evaluate" else ("profile", "patch")
        _keys(value, allowed, (), f"{label}.{phase}")
        if "profile" in value and (not isinstance(value["profile"], str) or not value["profile"]):
            raise ExperimentSpecError(f"{label}.{phase}.profile must be a nonempty path/name")
        if "patch" in value and not isinstance(value["patch"], dict):
            raise ExperimentSpecError(f"{label}.{phase}.patch must be an object")
        if "checkpoint_from" in value:
            source = value["checkpoint_from"]
            if (not isinstance(source, str) or source.startswith("/") or source.endswith("/") or
                    "//" in source or any(_SAFE.fullmatch(part) is None for part in source.split("/") if part)):
                raise ExperimentSpecError(f"{label}.{phase}.checkpoint_from must be a root-relative node path (empty for the root)")
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
        allowed.update(("schema_version", "kind", "execution"))
    _keys(raw, allowed, ("id",) if depth else ("schema_version", "kind", "id", "name"), path or "root")
    if depth == 0 and raw["schema_version"] != SCHEMA:
        raise ExperimentSpecError(f"schema_version must be {SCHEMA}")
    if depth == 0 and raw["kind"] != "workflow":
        raise ExperimentSpecError("kind must be workflow")
    if depth == 0 and "execution" in raw:
        _keys(raw["execution"], ("max_workers",), (), "execution")
        workers = raw["execution"].get("max_workers", _DEFAULT_WORKERS)
        if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
            raise ExperimentSpecError(f"execution.max_workers must be between 1 and {_MAX_WORKERS}")
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
        if "checkpoint_from" in value:
            current["checkpoint_from"] = value["checkpoint_from"]
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
        configuration = node.get("configuration", {}).get(phase, {})
        if phase == "evaluate" and "checkpoint_from" in configuration:
            source = (configuration["checkpoint_from"], "train")
            if source not in tasks:
                raise ExperimentSpecError(f"{path or '<root>'} checkpoint_from refers to an unselected/missing training phase {source}")
            edges[(path, phase)].add(source)
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


def build_experiment_index(plan, result, export_root: Path):
    from .experiment_overview import build_experiment_index as render
    return render(plan, result, export_root)


def _publish_experiment_index(plan, result, export_root: Path):
    from .experiment_overview import prepare_experiment_views
    prepare_experiment_views(plan, result, export_root)
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
        self.bundle = ConfigBundleBuilder(self.repository_root, environment=self.environment)
        self._process_lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()
        self._cancelled = threading.Event()

    def _spec_path(self, reference: str | Path) -> Path:
        if str(reference).startswith(("plugin:", "module:", "ovlab:")):
            return self.bundle.references.source(str(reference))
        path = Path(reference).expanduser()
        if path.is_absolute():
            return path.resolve()
        local = path.resolve()
        if local.is_file():
            return local
        direct = (self.repository_root / path).resolve()
        if direct.is_file():
            return direct
        names = {path.name} if path.suffix else {path.name + suffix for suffix in (".yaml", ".yml", ".json")}
        candidates = []
        for section in ("experiments/workflows", "experiments/training", "experiments/evaluation", "examples/experiments"):
            base = self.repository_root / section
            for candidate in base.rglob("*"):
                if candidate.name in names and ".generated" not in candidate.parts and candidate.is_file():
                    # Qualified section-relative names disambiguate equal basenames.
                    if len(path.parts) == 1 or candidate.relative_to(base).with_suffix("") == path.with_suffix(""):
                        candidates.append(candidate)
        if len(candidates) > 1:
            raise ExperimentSpecError(f"experiment name {str(reference)!r} is ambiguous; use a path: " + ", ".join(str(p) for p in sorted(candidates)))
        return candidates[0] if candidates else direct

    def _normalized_spec(self, path, raw):
        """Put standalone training/evaluation through the same tree owner."""
        if not isinstance(raw, dict):
            raise ExperimentSpecError("experiment must be an object")
        if raw.get("schema_version") == SCHEMA or raw.get("kind") == "workflow":
            return raw
        kind = raw.get("kind")
        if kind == "training":
            document = self._phase_document(path, "train")
            identifier, name, phase = document["id"], document["id"], "train"
        elif kind == "experiment":
            document = self._phase_document(path, "evaluate")
            identifier, name, phase = document["experiment"]["id"], document["experiment"]["name"], "evaluate"
        else:
            raise ExperimentSpecError("experiment kind must be workflow, training, or experiment (evaluation)")
        return {"schema_version": SCHEMA, "kind": "workflow", "id": identifier, "name": name,
                "config": {phase: {"profile": str(path)}}, "execute": [phase]}

    def _phase_document(self, path, phase):
        # Defaults may be partial. Validate the final document after workflow
        # patches, just as extends validates the final local configuration.
        if phase == "train":
            return load_training_document(path, expected_kind="training", repository_root=self.repository_root)
        try:
            reference = self.bundle.references.reference(path)
        except ValueError:
            return load_training_document(path, expected_kind="experiment", repository_root=self.repository_root)
        return self.bundle.load_component(reference, "experiment", validate_document=False)

    def _profile_path(self, reference, phase):
        if str(reference).startswith(("plugin:", "module:", "ovlab:")):
            candidate = self.bundle.references.source(str(reference))
        else:
            path = Path(reference).expanduser()
            if path.is_absolute():
                candidate = path.resolve()
            elif path.parts and path.parts[0] in {"configs", "config", "experiments", "examples"}:
                candidate = (self.repository_root / path).resolve()
            else:
                candidate = (self.repository_root / "experiments" / ("training" if phase == "train" else "evaluation") / path).resolve()
                if not candidate.is_file() and not candidate.suffix:
                    candidate = candidate.with_suffix(".yaml")
        if not candidate.is_file() or candidate.is_symlink():
            raise ExperimentSpecError(f"{phase} profile must be a regular file: {reference}")
        return candidate

    def _validate_component(self, document, reference, kind, *, checkpoint_template=False):
        if kind == "policy" and checkpoint_template and document.get("type") != "mock":
            from .schema import header, mapping
            from .policy_plugins import plugin_for_type, call_plugin
            header(document, reference, "policy", typed=True)
            mapping(document, reference, required=("schema_version", "kind", "type", "settings"), optional=("provider",))
            call_plugin(plugin_for_type(document["type"], self.repository_root, provider=document.get("provider")),
                        "validate_checkpoint_template", document, reference)
        else:
            validate(document, reference, kind, repository_root=self.repository_root)

    def _resolved_config(self, phase, state):
        path = self._profile_path(state["profile"], phase)
        try:
            reference = self.bundle.references.reference(path)
        except ValueError:
            reference = str(path)
        base = self._phase_document(path, phase)
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
        if "checkpoint_from" in state:
            result["checkpoint_from"] = state["checkpoint_from"]
        if phase == "evaluate":
            component_documents = {}
            for kind, patches in state["component_patches"].items():
                component_ref = document["components"][kind]
                component_base = self.bundle.load_component(component_ref, _COMPONENTS[kind], validate_document=False)
                component = deepcopy(component_base)
                for _, patch in patches:
                    component = _merge(component, patch)
                component.pop("extends", None)
                self._validate_component(component, component_ref, _COMPONENTS[kind],
                                         checkpoint_template="checkpoint_from" in state)
                component_documents[kind] = {
                    "profile": component_ref,
                    "document": component,
                    "field_sources": _field_sources(component_base, component_ref, patches),
                }
            result["component_documents"] = component_documents
            # Verify every unpatched reference without requiring runtime resources.
            for kind, component_ref in document["components"].items():
                if kind not in component_documents:
                    component = self.bundle.load_component(component_ref, _COMPONENTS[kind], validate_document=False)
                    self._validate_component(component, component_ref, _COMPONENTS[kind],
                                             checkpoint_template="checkpoint_from" in state)
                    if kind == "policy" and "checkpoint_from" in state:
                        component_documents[kind] = {
                            "profile": component_ref, "document": component,
                            "field_sources": _field_sources(component, component_ref, []),
                        }
            self.bundle.load_component(document["resources"]["registry"], "resource_registry")
        return result

    @with_registry
    def plan(self, spec: str | Path, *, node: str | None = None, node_only=False,
             sequential=False, max_workers=None):
        if sequential and max_workers is not None:
            raise ExperimentSpecError("--sequential and --max-workers cannot be combined")
        spec_path = self._spec_path(spec)
        source = _read_spec(spec_path)
        raw = self._normalized_spec(spec_path, source)
        _validate_node(raw)
        workers = 1 if sequential else (raw.get("execution", {}).get("max_workers", _DEFAULT_WORKERS) if max_workers is None else max_workers)
        if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
            raise ExperimentSpecError(f"max workers must be between 1 and {_MAX_WORKERS}")
        if node is not None and node and any(_SAFE.fullmatch(part) is None for part in node.split("/")):
            raise ExperimentSpecError("--node must be a slash-separated path of node IDs")
        root_id = _safe_id(raw["id"], "root.id")
        spec_hash = _spec_hash(source)
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
            nodes.append({
                "id": item["id"], "name": item.get("name", item["id"]), "path": path,
                "parent": path.rpartition("/")[0] if path else None,
                "execute": list(execute), "depends_on": deepcopy(item.get("depends_on", [])),
                "configuration": configuration,
                "children": [child["id"] for child in item.get("children", [])],
            })
        steps = _topological_steps(nodes, selected)
        by_path = {item["path"]: item for item in nodes}
        for item in nodes:
            evaluation = item["configuration"].get("evaluate", {})
            if item['path'] not in selected or "evaluate" not in item["execute"] or "checkpoint_from" not in evaluation:
                continue
            training = by_path[evaluation["checkpoint_from"]]["configuration"]["train"]["document"]
            if "deployment" not in evaluation['document']:
                raise ExperimentSpecError('checkpoint_from requires a deployed evaluation policy')
            from .workflow_checkpoints import validate_checkpoint_binding
            policy = evaluation.get("component_documents", {}).get("policy", {}).get("document")
            if policy is None:
                policy = self.bundle.load_component(evaluation["document"]["components"]["policy"], "policy")
            validate_checkpoint_binding(training, policy)
        return {
            "schema_version": PLAN_SCHEMA, "spec_schema_version": source.get("schema_version"),
            "source_kind": source.get("kind"),
            "spec": str(spec_path), "spec_hash": spec_hash, "id": root_id,
            "name": raw["name"], "selected_node": selected_path,
            "node_only": bool(node_only), "nodes": nodes, "steps": steps,
            "selected_nodes": sorted(selected),
            "execution": {"mode": "sequential" if workers == 1 else "parallel", "max_workers": workers},
            "dependency_semantics": "depends_on schedules phases; checkpoint_from binds a finalized training checkpoint and adds its training dependency",
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
            if "registry_document" in configuration:
                registry = self.config_root / "resources" / ".generated" / instance
                registry = registry.joinpath(*parts[:-1], parts[-1] + ".yaml")
                registry.parent.mkdir(parents=True, exist_ok=True)
                registry.write_text(dumps(configuration["registry_document"]), encoding="utf-8")
                document["resources"]["registry"] = registry.relative_to(self.config_root).as_posix()
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
            env = dict(
                self.environment, OVLAB_RUNS_ROOT=str(runs_root), OVLAB_EXPORTS_ROOT=str(exports_root),
                OVLAB_EXPERIMENT_ID=instance, OVLAB_EXPERIMENT_NODE=node_path,
            )
            binding = None
            if phase == "evaluate" and "checkpoint_from" in configuration:
                from .workflow_checkpoints import bind_training_checkpoint
                configuration, binding, overrides = bind_training_checkpoint(
                    self.repository_root, self.environment, configuration,
                    item.get("_checkpoint_record"), runs_root=runs_root,
                    experiment_id=instance, bundle=self.bundle,
                    expected_profile=item.get("_checkpoint_training_profile"),
                )
                env.update(overrides)
                binding_path = runs_root / instance / ".experiment" / "checkpoint-bindings"
                parts = ["nodes", *node_path.split("/")] if node_path else ["root"]
                binding_path = binding_path.joinpath(*parts[:-1], parts[-1] + ".json")
                _write_json(binding_path, {"binding": binding, "configuration": configuration})
            generated = self._materialize(instance, node_path, phase, configuration)
            if phase == "train":
                command = ["train", "run", str(generated)]
            elif "deployment" in configuration["document"]:
                command = ["eval", "run", str(generated)]
            else:
                command = ["_native-run", str(generated)]
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
            if binding is not None:
                record["checkpoint_binding"] = binding
            if phase == "train":
                record["isolated_export"] = self._invoke([
                    "export", "isolated", "--category", "train", "--run", run_id,
                ], env)
            elif command[0] == "_native-run":
                reporting = configuration["document"].get("reporting", {})
                profile = reporting.get("profile", "libero-task-default")
                record["postprocessing"] = self._invoke([
                    "report", "publish", "--run", run_id, "--profile", profile,
                    "--report-enabled", "true" if reporting.get("enabled", True) else "false",
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
                                item = deepcopy(nodes[node_path])
                                configuration = item["configuration"][phase]
                                if phase == "evaluate" and "checkpoint_from" in configuration:
                                    source = configuration["checkpoint_from"]
                                    item["_checkpoint_record"] = deepcopy(result.get("nodes", {}).get(source, {}).get("train"))
                                    item["_checkpoint_training_profile"] = deepcopy(nodes[source]["configuration"]["train"]["document"])
                                future = executor.submit(self._execute_step, result["experiment_id"],
                                                         node_path, phase, item, runs_root, exports_root)
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

    def _verify_run(self, path, phase):
        if phase == "train":
            from .training_runs import TrainingRunStore
            data_root = Path(self.environment.get("OVLAB_MODEL_DATA_ROOT", self.environment.get(
                "OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data")))
            return TrainingRunStore(data_root, training_runs_root=path.parent).verify(path.name)
        from ovlab_runner import verify_run
        return verify_run(path)

    def verify(self, reference: str | Path):
        """Validate a source configuration or check all recorded canonical runs."""
        from .experiment_references import find_experiment
        from .run_references import resolve_run_reference_tree
        source = self._spec_path(reference)
        if source.is_file():
            plan = self.plan(source)
            return {"schema_version": "ovlab.experiment-verification/v1", "status": "verified",
                    "source_type": "configuration", "id": plan["id"], "spec": plan["spec"],
                    "spec_hash": plan["spec_hash"], "nodes": plan["nodes"], "steps": plan["steps"],
                    "side_effects_performed": False}
        data_root = Path(self.environment.get("OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data"))
        runs_root = Path(self.environment.get("OVLAB_RUNS_ROOT", data_root / "runs")).expanduser().resolve()
        experiment = find_experiment(runs_root, str(reference))
        if experiment is None:
            run = resolve_run_reference_tree(runs_root, str(reference))
            phase = "train" if (run.path / "training-plan.json").is_file() else "evaluate"
            checked = self._verify_run(run.path, phase)
            return {"schema_version": "ovlab.experiment-verification/v1", "status": "verified",
                    "source_type": "run", "id": run.run_id, "verified_runs": [checked],
                    "side_effects_performed": False}
        documents = {}
        for name in ("spec", "plan", "result"):
            path = experiment / ".experiment" / (name + ".json")
            if path.is_symlink() or not path.is_file():
                raise ExperimentIntegrityError(f"experiment metadata is unavailable or a symbolic link: {path}")
            try:
                document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
            except (OSError, ValueError, ExperimentSpecError) as exc:
                raise ExperimentIntegrityError(f"cannot read experiment metadata: {path}") from exc
            if not isinstance(document, dict):
                raise ExperimentIntegrityError(f"experiment metadata must be an object: {path}")
            documents[name] = document
        spec, plan, result = (documents[name] for name in ("spec", "plan", "result"))
        migrated = plan.get("migration_only") is True
        if (plan.get("schema_version") != PLAN_SCHEMA or result.get("schema_version") != RESULT_SCHEMA or
                plan.get("experiment_id") != experiment.name or result.get("experiment_id") != experiment.name):
            raise ExperimentIntegrityError("experiment metadata identity or specification hash does not match")
        if migrated:
            path = experiment / ".experiment" / "migration.json"
            if path.is_symlink() or not path.is_file():
                raise ExperimentIntegrityError("migration record is unavailable or a symbolic link")
            try:
                migration = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
            except (OSError, ValueError, ExperimentSpecError) as exc:
                raise ExperimentIntegrityError("invalid migration record") from exc
            if (not isinstance(migration, dict) or migration != result.get("migration") or
                    migration.get("schema_version") != "ovlab.experiment-migration/v1" or
                    spec.get("id") != experiment.name or not isinstance(result.get("nodes"), dict) or
                    set(result["nodes"]) != {""} or not isinstance(result["nodes"][""], dict) or
                    len(result["nodes"][""]) != 1):
                raise ExperimentIntegrityError("migration metadata is inconsistent")
            steps = [{"node": "", "phase": phase} for phase in result["nodes"][""]]
        else:
            if plan.get("spec_hash") != _spec_hash(spec) or result.get("spec_hash") != plan.get("spec_hash"):
                raise ExperimentIntegrityError("experiment specification hash does not match")
            steps = plan.get("steps")
        if result.get("status") != ("migrated" if migrated else "completed"):
            raise ExperimentIntegrityError(f"experiment {experiment.name!r} is not completed ({result.get('status', 'unknown')})")
        if not isinstance(steps, list) or not steps or not isinstance(result.get("nodes"), dict):
            raise ExperimentIntegrityError("experiment has no valid phase inventory")
        checked, expected = [], set()
        if not isinstance(plan.get('nodes'), list) or any(not isinstance(node, dict) or not isinstance(node.get('path'), str) for node in plan['nodes']):
            raise ExperimentIntegrityError('experiment has an invalid node inventory')
        planned_nodes = {node['path']: node for node in plan['nodes']}
        verified_checkpoints = set()
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("node"), str) or step.get("phase") not in _PHASES:
                raise ExperimentIntegrityError("invalid phase in experiment plan")
            node, phase = step["node"], step["phase"]
            key = (node, phase)
            if key in expected:
                raise ExperimentIntegrityError("duplicate phase in experiment plan")
            expected.add(key)
            records = result["nodes"].get(node, {})
            record = records.get(phase) if isinstance(records, dict) else None
            if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
                raise ExperimentIntegrityError(f"missing run for {node or '<root>'} {phase}")
            run_id = record["run_id"]
            if not _SAFE.fullmatch(run_id):
                raise ExperimentIntegrityError("invalid recorded run ID")
            try:
                scope = ExperimentScope(experiment.name, tuple(node.split("/")) if node else ())
                path = scope.runs_root(runs_root, phase) / run_id
                resolved = resolve_run_reference_tree(runs_root, str(path), category=phase)
            except (ValueError, RuntimeError) as exc:
                raise ExperimentIntegrityError(f"invalid or missing run for {node or '<root>'} {phase}: {exc}") from exc
            configured = planned_nodes.get(node, {}).get('configuration', {}).get(phase, {})
            if phase == 'evaluate' and 'checkpoint_from' in configured:
                from .workflow_checkpoints import verify_saved_binding
                proof = record.get('checkpoint_binding')
                parts = ['nodes', *node.split('/')] if node else ['root']
                binding_file = experiment / '.experiment/checkpoint-bindings'
                binding_file = binding_file.joinpath(*parts[:-1], parts[-1] + '.json')
                try:
                    verify_saved_binding(self.repository_root, self.environment, configured, proof,
                                         result['nodes'], runs_root, experiment.name, binding_file,
                                         verified_checkpoints, planned_nodes[configured['checkpoint_from']]['configuration']['train']['document'])
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                    raise ExperimentIntegrityError(f'invalid checkpoint binding for {node or "<root>"}: {exc}') from exc
            verification = self._verify_run(resolved.path, phase)
            if verification.get("status") not in {"completed", "verified"}:
                raise ExperimentIntegrityError(f"recorded phase is not completed: {node or '<root>'} {phase}")
            if migrated:
                from .experiment_migration import _inventory
                if _inventory(resolved.path) != migration.get("source_inventory"):
                    raise ExperimentIntegrityError("migrated run no longer matches its original inventory")
            checked.append({"node": node, "phase": phase, **verification})
        actual = set()
        for node, records in result["nodes"].items():
            if not isinstance(records, dict):
                raise ExperimentIntegrityError("invalid experiment result node")
            actual.update((node, phase) for phase in records)
        if actual != expected:
            raise ExperimentIntegrityError("experiment result phases do not match its plan")
        return {"schema_version": "ovlab.experiment-verification/v1", "status": "verified",
                "source_type": "experiment", "experiment_id": experiment.name,
                "spec_hash": _spec_hash(spec), "verified_runs": checked, "side_effects_performed": False}

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
            for section in ("experiments/training", "experiments/evaluation", "benchmarks", "policies", "metrics", "protocols", "artifacts", "resources"):
                generated_root = (self.repository_root if section.startswith("experiments/") else self.config_root) / section / ".generated" / instance
                if generated_root.is_dir():
                    shutil.rmtree(generated_root)
                    try:
                        generated_root.parent.rmdir()
                    except OSError:
                        pass
