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


def _catalog(root):
    from .extensions import ExtensionCatalog
    from .plugin_projects import read_document
    from .policy_plugins import registry_path
    path = registry_path(root)
    if read_document(path).get("schema_version") == "ovlab.extensions/v1":
        return ExtensionCatalog(root, registry_path=path)
    return None


def list_projects(root, *, category="plugin"):
    catalog = _catalog(root)
    if catalog is not None:
        rows = []
        for row in catalog.list_projects():
            if row.get("category") not in {category, None}:
                continue
            row["kind"] = "unknown"
            row["variants"] = []
            if row.get("status") == "available":
                manifest = catalog.project_manifest(row["id"])
                row["kind"] = ",".join(dict.fromkeys(item["kind"] for item in manifest["providers"].values())) or "configuration"
                row["variants"] = [item.get("id", name) for name, item in manifest["providers"].items()]
            rows.append(row)
        return rows
    if category != "plugin":
        return []
    document = registry_document(root)
    return [{"id": name, "kind": metadata.get("kind", "policy"), "version": metadata["version"],
             "source": deepcopy(document["sources"][metadata["source_id"]]),
             "variants": [item["id"] for item in document["plugins"].values() if item.get("project") == name]}
            for name, metadata in document.get("projects", {}).items()]


def inspect_project(root, name, *, category="plugin"):
    catalog = _catalog(root)
    if catalog is not None:
        manifest = catalog.project_manifest(name)
        if manifest["category"] != category:
            raise ConfigReferenceError(f"project {name} is not a {category}")
        metadata = catalog.project_metadata(name)
        variants = [{"id": local_id, **deepcopy(item), "provider_ref": f"{category}:{item['kind']}:{name}/{local_id}"}
                    for local_id, item in manifest["providers"].items()]
        return {**metadata, "kind": ",".join(dict.fromkeys(item["kind"] for item in variants)) or "configuration",
                "source": deepcopy(catalog.document["projects"][name]["source"]),
                "variants": variants, "sources": [{"id": key, "selection": value}
                                                       for key, value in catalog.project_sources(name).items()]}
    document = registry_document(root)
    variants = project_plugins(root, name, registry=document)
    metadata = document["projects"][name]
    return {"id": name, **deepcopy(metadata), "kind": metadata.get("kind", "policy"),
            "source": deepcopy(document["sources"][metadata["source_id"]]),
            "variants": [deepcopy(plugin.descriptor) for plugin in variants],
            "sources": inspect_sources(root, project_sources(document, name), registry=document)}


def check_project(root, name, *, category="plugin"):
    from .strict_yaml import load
    catalog = _catalog(root)
    if catalog is not None:
        info = inspect_project(root, name, category=category)
        checked_templates = []
        for paths in info["config_templates"].values():
            for relative in paths:
                reference = f"{category}://{name}/{relative}"
                path = catalog.config_path(reference)
                if not path.is_file():
                    raise ConfigReferenceError(f"{category} {name} is missing its configuration template: {relative}")
                load(path)
                checked_templates.append(reference)
        variants = []
        for item in info["variants"]:
            if item["kind"] == "policy":
                from ovlab_policy_sdk.plugin import PolicyPlugin
                contract = PolicyPlugin
            elif item["kind"] == "benchmark":
                from ovlab_benchmarks import BenchmarkPlugin
                contract = BenchmarkPlugin
            elif item["kind"] == "dataset":
                from ovlab_core.datasets import DatasetProvider
                contract = DatasetProvider
            else:
                raise ConfigCompatibilityError(f"SDK control checking is not supported yet for {category}/{item['kind']}")
            implementation = catalog.load_provider(item["provider_ref"], expected_kind=item["kind"], api_version=item["api_version"], contract=contract)
            variants.append({"id": item["id"], "kind": item["kind"], "config_type": item.get("config_type"), "api_version": implementation.api_version})
        return {"id": name, "valid": True, "category": category, "kind": info["kind"],
                "checks": ["manifest", "configuration_templates", *(["sdk_control_api"] if variants else [])],
                "templates": checked_templates, "variants": variants, "runtime_tested": False,
                "model_loaded": False}
    document = registry_document(root)
    plugins = project_plugins(root, name, registry=document)
    metadata = document["projects"][name]
    if metadata.get("kind", "policy") == "policy" and not plugins:
        raise ConfigCompatibilityError(f"policy plugin {name} declares no policy variants")
    project = project_root(name, root, document)
    configs = _inside(project, metadata["configs_root"], "plugin configuration root")
    checked_templates = []
    for template_category, paths in metadata.get("config_templates", {}).items():
        for relative in paths:
            path = _inside(configs, relative, "plugin configuration template")
            if not path.is_file():
                raise ConfigReferenceError(f"plugin {name} is missing its {template_category} template: {relative}")
            load(path)
            checked_templates.append(f"plugin://{name}/{relative}")
    variants = [{"id": plugin.descriptor["id"], "config_type": plugin.descriptor["config_type"],
                 "api_version": plugin.implementation.api_version} for plugin in plugins]
    return {"id": name, "valid": True, "kind": metadata.get("kind", "policy"),
            "checks": ["manifest", "configuration_templates", "sdk_control_api"],
            "templates": checked_templates, "variants": variants, "runtime_tested": False, "model_loaded": False}
