"""Project-level plugin inspection and control API checks without model loading."""
from copy import deepcopy

from .errors import ConfigCompatibilityError, ConfigReferenceError
from .plugin_projects import _inside, project_root
from .plugin_sources import inspect_sources
from .policy_plugins import LoadedPolicyPlugin, registry_document


def project_plugins(root, project, *, registry=None):
    document = registry if registry is not None else registry_document(root)
    if project not in document.get("projects", {}):
        raise ConfigReferenceError(f"unknown plugin project: {project}")
    return [LoadedPolicyPlugin(deepcopy(descriptor), root, document)
            for descriptor in document["plugins"].values() if descriptor.get("project") == project]


def project_roles(document, project=None):
    """Select complete project builds, excluding infrastructure recipes."""
    projects = document.get("projects", {})
    if project is not None and project not in projects:
        raise ConfigReferenceError(f"unknown plugin project: {project}")
    selected = {project} if project is not None else set(projects)
    return [name for name, recipe in document["builds"].items() if recipe.get("project") in selected]


def project_sources(document, project):
    if project not in document.get("projects", {}):
        raise ConfigReferenceError(f"unknown plugin project: {project}")
    sources = [document["projects"][project]["source_id"]]
    for variant in document["plugins"].values():
        if variant.get("project") == project:
            sources.extend(variant.get("sources", []))
    for role in project_roles(document, project):
        sources.extend(document["builds"][role].get("sources", []))
    return list(dict.fromkeys(sources))


def list_projects(root):
    document = registry_document(root)
    return [{
        "id": name, "kind": metadata.get("kind", "policy"), "version": metadata["version"],
        "source": deepcopy(document["sources"][metadata["source_id"]]),
        "variants": [item["id"] for item in document["plugins"].values() if item.get("project") == name],
    } for name, metadata in document.get("projects", {}).items()]


def inspect_project(root, name):
    document = registry_document(root)
    variants = project_plugins(root, name, registry=document)
    metadata = document["projects"][name]
    return {"id": name, **deepcopy(metadata), "kind": metadata.get("kind", "policy"),
            "source": deepcopy(document["sources"][metadata["source_id"]]),
            "variants": [deepcopy(plugin.descriptor) for plugin in variants],
            "sources": inspect_sources(root, project_sources(document, name), registry=document)}


def check_project(root, name):
    document = registry_document(root)
    plugins = project_plugins(root, name, registry=document)
    metadata = document["projects"][name]
    if metadata.get("kind", "policy") == "policy" and not plugins:
        raise ConfigCompatibilityError(f"policy plugin {name} declares no policy variants")
    project = project_root(name, root, document)
    configs = _inside(project, metadata["configs_root"], "plugin configuration root")
    from .strict_yaml import load
    checked_templates = []
    for category, paths in metadata.get("config_templates", {}).items():
        for relative in paths:
            path = _inside(configs, relative, "plugin configuration template")
            if not path.is_file():
                raise ConfigReferenceError(f"plugin {name} is missing its {category} template: {relative}")
            load(path)
            checked_templates.append(f"plugin://{name}/{relative}")
    variants = [{"id": plugin.descriptor["id"], "config_type": plugin.descriptor["config_type"],
                 "api_version": plugin.implementation.api_version} for plugin in plugins]
    return {"id": name, "valid": True, "kind": metadata.get("kind", "policy"),
            "checks": ["manifest", "configuration_templates", "sdk_control_api"],
            "templates": checked_templates, "variants": variants,
            "runtime_tested": False, "model_loaded": False}
