"""Registry input for the plugin's standalone diagnostic service entrypoints."""
from copy import deepcopy
from pathlib import Path

from ovlab_policy_sdk.strict_yaml import load


def _merge(parent, child):
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_resource_registry(path, *, _root=None, _stack=()):
    """Read a bounded registry for diagnostics without depending on the OVLAB CLI.

    Production uses the generic OVLAB service with already validated config.
    Standalone services accept the same registry inheritance inside its directory.
    Model artifact constructors validate the selected checkpoint's identity.
    """
    path = Path(path).resolve()
    root = _root or path.parent
    if not path.is_relative_to(root) or path in _stack:
        raise ValueError("resource registry inheritance escapes its root or contains a cycle")
    document = load(path)
    if not isinstance(document, dict) or document.get("kind") != "resource_registry":
        raise ValueError("expected a resource_registry document")
    if document.get("schema_version") != "0.1.0":
        raise ValueError("unsupported resource registry schema_version")
    if "extends" in document:
        reference = document["extends"]
        if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
            raise ValueError("resource registry extends must be a relative path")
        parent = load_resource_registry(path.parent / reference, _root=root, _stack=(*_stack, path))
        document = _merge(parent, document)
    if not isinstance(document.get("checkpoints"), dict):
        raise ValueError("resource registry checkpoints must be a mapping")
    return document
