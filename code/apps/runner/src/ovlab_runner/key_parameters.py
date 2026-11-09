"""Resolve template-selected parameters from recorded report evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import importlib.resources
import json
import math
from pathlib import Path
import re

from .errors import ArtifactError


SCHEMA = "ovlab.key-parameters/v1"
_STAGES = {"inputs", "representation", "policy", "outputs", "result"}
_SOURCES = {
    "evaluate": {"plan.json", "connection.json", "report.json"},
    "train": {"training-plan.json", "result.json", "manifest.json", "data-flow.json"},
}
_PATH = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")


def validate_key_parameters(document, mode: str) -> dict[str, object]:
    """Validate declarative lookups; paths never access files or execute code."""
    if mode not in _SOURCES or not isinstance(document, Mapping):
        raise ArtifactError("key parameters must be a configuration object for train or evaluate")
    if set(document) != {"schema_version", "mode", "groups"} or document.get("schema_version") != SCHEMA or document.get("mode") != mode:
        raise ArtifactError(f"key parameters must use {SCHEMA} and mode {mode}")
    groups = document["groups"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= len(_STAGES):
        raise ArtifactError("key parameter groups must be a non-empty list of distinct stages")
    stages, labels = set(), set()
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"title", "stage", "items"}:
            raise ArtifactError("each key parameter group must contain title, stage and items")
        title, stage, items = group["title"], group["stage"], group["items"]
        if not isinstance(title, str) or not title.strip() or not isinstance(stage, str) or stage not in _STAGES or stage in stages:
            raise ArtifactError("key parameter groups require a title and a distinct supported stage")
        stages.add(stage)
        if not isinstance(items, list) or not 1 <= len(items) <= 100:
            raise ArtifactError("key parameter group items must be a non-empty list of at most 100 items")
        for item in items:
            if not isinstance(item, Mapping) or set(item) != {"label", "sources"}:
                raise ArtifactError("each key parameter item must contain label and sources")
            label, candidates = item["label"], item["sources"]
            if not isinstance(label, str) or not label.strip() or label in labels:
                raise ArtifactError("key parameter labels must be non-empty and unique")
            labels.add(label)
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20:
                raise ArtifactError("key parameter sources must be a non-empty list of at most 20 lookups")
            for candidate in candidates:
                if (not isinstance(candidate, Mapping) or set(candidate) != {"source", "path"}
                        or not isinstance(candidate.get("source"), str) or candidate["source"] not in _SOURCES[mode]
                        or not isinstance(candidate.get("path"), str)
                        or _PATH.fullmatch(candidate["path"]) is None):
                    raise ArtifactError(f"invalid key parameter source or dotted path for {mode}")
    return deepcopy(dict(document))


def _json_document(text: str, label: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid key parameter JSON in {label}: {exc}") from exc


def load_key_parameters(path: str | Path, mode: str) -> dict[str, object]:
    path = Path(path).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"cannot read key parameter configuration: {path}") from exc
    return validate_key_parameters(_json_document(text, str(path)), mode)


def default_key_parameters(mode: str) -> dict[str, object]:
    if mode not in _SOURCES:
        raise ArtifactError(f"unknown key parameter mode: {mode}")
    resource = importlib.resources.files("ovlab_runner").joinpath(
        f"report_assets/profiles/{mode}-key-parameters.json"
    )
    return validate_key_parameters(_json_document(resource.read_text(encoding="utf-8"), str(resource)), mode)


def resolve_key_parameters(configuration, mode: str, sources: Mapping) -> list[dict[str, object]]:
    config = validate_key_parameters(configuration, mode)
    groups = []
    for group in config["groups"]:
        items = []
        for item in group["items"]:
            resolved = {"label": item["label"], "value": None, "source": None}
            for candidate in item["sources"]:
                name, path = candidate["source"], candidate["path"]
                value = sources.get(name)
                for key in path.split("."):
                    if not isinstance(value, Mapping) or key not in value:
                        value = None
                        break
                    value = value[key]
                if (isinstance(value, (str, int, float, bool)) and value != ""
                        and (not isinstance(value, float) or math.isfinite(value))):
                    resolved.update(value=value, source=f"{name if name == 'report.json' else 'run/' + name}:{path}")
                    break
            items.append(resolved)
        groups.append({"title": group["title"], "stage": group["stage"], "items": items})
    return groups


def evaluation_parameters(plan, connection, complexity, configuration=None):
    sources = {"plan.json": plan, "connection.json": connection,
               "report.json": {"performance": {"model_complexity": complexity}}}
    return resolve_key_parameters(configuration if configuration is not None else default_key_parameters("evaluate"), "evaluate", sources)


def training_parameters(plan, result, manifest, evidence=None, configuration=None):
    sources = {"training-plan.json": plan, "result.json": result,
               "manifest.json": manifest, "data-flow.json": evidence or {}}
    return resolve_key_parameters(configuration if configuration is not None else default_key_parameters("train"), "train", sources)
