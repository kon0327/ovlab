"""Project selection stays declarative, isolated and independent of host paths.

These tests qualify registry/selected-lock behavior. They do not qualify a
benchmark runtime, the public evaluation pipeline, Docker or GPU execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ovlab_benchctl.errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from ovlab_benchctl.extensions import ExtensionCatalog, ProviderRef


REMOTE_COMMIT = "b" * 40
POLICY_REF = "plugin:policy:isolation-policy/default"


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _project(path: Path, *, category: str, identifier: str) -> Path:
    path.mkdir(parents=True)
    (path / "configs").mkdir()
    (path / "configs/sample.yaml").write_text(
        "schema_version: '1'\nkind: fixture_config\nvalue: preserved\n", encoding="utf-8"
    )
    providers = {}
    if category == "plugin":
        providers["default"] = {
            "kind": "policy",
            "api_version": "1",
            "config_type": "isolation_policy",
            "entrypoint": "isolation_control:plugin",
            "runtime_status": "implemented",
            "sources": [],
        }
        package = path / "src/isolation_control"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "raise AssertionError('declarative selection imported plugin code')\n", encoding="utf-8"
        )
    _write_json(path / "extension.json", {
        "schema_version": "ovlab.extension/v1",
        "id": identifier,
        "category": category,
        "version": "0.0.0",
        "providers": providers,
        "configs": {"root": "configs", "templates": {"sample": ["sample.yaml"]}},
    })
    return path


def _registry(root: Path, projects: dict) -> ExtensionCatalog:
    document = {"schema_version": "ovlab.extensions/v1", "projects": projects}
    path = root / "configs/extensions/registry.json"
    _write_json(path, document)
    return ExtensionCatalog(root, registry_path=path)


def _local(path: Path) -> dict:
    return {"source": {"kind": "local", "path": str(path)}}


def _remote() -> dict:
    return {"source": {"kind": "git", "url": "https://unused.invalid/never-fetch.git", "commit": REMOTE_COMMIT}}


def _tree(path: Path) -> tuple:
    return tuple(
        (item.relative_to(path).as_posix(), item.read_bytes() if item.is_file() else None)
        for item in sorted(path.rglob("*"))
    )


def _forbid_processes(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail(f"declarative registry operation started a process: {args[0]!r}")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def test_listing_uncached_remote_project_does_not_fetch_import_or_write(tmp_path, monkeypatch):
    root = tmp_path / "ovlab"
    policy = _project(tmp_path / "independent-policy", category="plugin", identifier="isolation-policy")
    catalog = _registry(root, {"isolation-policy": _local(policy), "unused-remote": _remote()})
    before = _tree(tmp_path)
    _forbid_processes(monkeypatch)

    rows = {row["id"]: row for row in catalog.list_projects()}

    assert rows["unused-remote"]["status"] == "unavailable"
    assert "isolation-policy" in rows
    assert "isolation_control" not in sys.modules
    assert _tree(tmp_path) == before


def test_explicit_provider_selection_ignores_unused_remote_project(tmp_path, monkeypatch):
    root = tmp_path / "ovlab"
    policy = _project(tmp_path / "independent-policy", category="plugin", identifier="isolation-policy")
    catalog = _registry(root, {"isolation-policy": _local(policy), "unused-remote": _remote()})
    before = _tree(tmp_path)
    _forbid_processes(monkeypatch)

    selected = catalog.resolve_provider(POLICY_REF, expected_kind="policy")
    snapshot = catalog.snapshot(provider_refs=(POLICY_REF,))

    assert str(selected.ref) == POLICY_REF
    assert selected.project_root == policy.resolve()
    assert selected.descriptor["kind"] == "policy"
    assert "unused-remote" not in json.dumps(snapshot.lock, sort_keys=True)
    assert "isolation_control" not in sys.modules
    assert _tree(tmp_path) == before


def test_runtime_projection_ignores_invalid_unselected_project(tmp_path):
    from ovlab_benchctl.policy_plugins import registry_document

    policy = _project(tmp_path / 'policy', category='plugin', identifier='isolation-policy')
    valid = json.loads((policy / 'extension.json').read_text())
    valid['providers']['default']['family'] = 'isolation'
    _write_json(policy / 'extension.json', valid)
    broken = _project(tmp_path / 'broken', category='plugin', identifier='broken')
    manifest = json.loads((broken / 'extension.json').read_text())
    manifest['providers']['default']['entrypoint'] = 'invalid-entrypoint'
    _write_json(broken / 'extension.json', manifest)
    root = tmp_path / 'host'
    catalog = _registry(root, {'isolation-policy': _local(policy), 'broken': _local(broken)})

    projected = registry_document(root)
    assert set(projected['plugins']) == {POLICY_REF}
    assert set(projected['projects']) == {'isolation-policy'}
    assert set(catalog.config_roots()) == {'isolation-policy'}
    assert catalog.resolve_provider(POLICY_REF).ref.project == 'isolation-policy'
    with pytest.raises(ConfigSchemaError, match='entrypoint'):
        catalog.resolve_provider('plugin:policy:broken/default')


def test_provider_reference_retains_category_kind_project_and_provider():
    reference = ProviderRef("plugin", "policy", "isolation-policy", "default")
    assert str(reference) == POLICY_REF


def test_provider_selection_rejects_a_different_expected_kind(tmp_path):
    policy = _project(tmp_path / "independent-policy", category="plugin", identifier="isolation-policy")
    catalog = _registry(tmp_path / "ovlab", {"isolation-policy": _local(policy)})

    with pytest.raises((ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError)):
        catalog.resolve_provider(POLICY_REF, expected_kind="benchmark")


def test_module_configuration_needs_no_provider_or_runtime_import(tmp_path, monkeypatch):
    module = _project(tmp_path / "independent-module", category="module", identifier="fixture-module")
    catalog = _registry(tmp_path / "ovlab", {"fixture-module": _local(module), "unused-remote": _remote()})
    before = _tree(tmp_path)
    _forbid_processes(monkeypatch)

    manifest = catalog.project_manifest("fixture-module")
    selected = catalog.config_path("module://fixture-module/sample.yaml")
    snapshot = catalog.snapshot(config_refs=("module://fixture-module/sample.yaml",))

    assert manifest["category"] == "module" and manifest["providers"] == {}
    assert selected == (module / "configs/sample.yaml").resolve()
    assert "value: preserved" in selected.read_text(encoding="utf-8")
    assert "fixture-module" in json.dumps(snapshot.lock, sort_keys=True)
    assert "unused-remote" not in json.dumps(snapshot.lock, sort_keys=True)
    assert _tree(tmp_path) == before


def test_configuration_reference_cannot_relabel_project_category(tmp_path):
    module = _project(tmp_path / "independent-module", category="module", identifier="fixture-module")
    catalog = _registry(tmp_path / "ovlab", {"fixture-module": _local(module)})

    with pytest.raises((ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError)):
        catalog.config_path("plugin://fixture-module/sample.yaml")


def test_selected_local_identity_survives_relocation_and_changes_with_content(tmp_path):
    original = _project(tmp_path / "first-location/module", category="module", identifier="fixture-module")
    relocated = tmp_path / "different-location/module"
    shutil.copytree(original, relocated)
    first = _registry(tmp_path / "first-host", {"fixture-module": _local(original)})
    second = _registry(tmp_path / "second-host", {"fixture-module": _local(relocated)})

    original_snapshot = first.snapshot(config_refs=("module://fixture-module/sample.yaml",))
    relocated_snapshot = second.snapshot(config_refs=("module://fixture-module/sample.yaml",))

    assert original_snapshot.lock == relocated_snapshot.lock
    assert original_snapshot.sha256 == relocated_snapshot.sha256
    assert len(original_snapshot.sha256) == 64
    assert original_snapshot.locations != relocated_snapshot.locations
    portable = json.dumps(original_snapshot.lock, sort_keys=True)
    assert str(tmp_path) not in portable
    (relocated / "configs/sample.yaml").write_text("kind: fixture_config\nvalue: changed\n", encoding="utf-8")
    changed = second.snapshot(config_refs=("module://fixture-module/sample.yaml",))
    assert changed.sha256 != relocated_snapshot.sha256


@pytest.mark.parametrize("reference", [
    "module://fixture-module/../extension.json",
    "module://fixture-module/%2e%2e/extension.json",
    "module://fixture-module/sample.yaml?override=1",
    "module://fixture-module/sample.yaml#other",
])
def test_module_configuration_rejects_ambiguous_or_escaping_paths(tmp_path, reference):
    module = _project(tmp_path / "independent-module", category="module", identifier="fixture-module")
    catalog = _registry(tmp_path / "ovlab", {"fixture-module": _local(module)})

    with pytest.raises((ConfigReferenceError, ConfigSchemaError)):
        catalog.config_path(reference)


def test_runtime_projection_preserves_equal_local_source_and_build_names(tmp_path):
    from ovlab_benchctl.policy_plugins import registry_document
    projects = {}
    for name in ('first', 'second'):
        source = _project(tmp_path / name, category='plugin', identifier=name)
        (source / 'Dockerfile').write_text('FROM scratch\n')
        (source / 'upstream').mkdir()
        (source / 'upstream/implementation.py').write_text(f'identity = {name!r}\n')
        manifest = json.loads((source / 'extension.json').read_text())
        manifest['providers'] = {'native': {'kind':'benchmark', 'api_version':'1',
            'entrypoint':'fixture:plugin', 'config_type':'shared_type', 'sources':['upstream'],
            'deployment':{'role':'benchmark', 'image':f'fixture-{name}:local', 'image_env':'OVLAB_BENCHMARK_IMAGE'}}}
        manifest['sources'] = {'upstream': {'kind':'local', 'path':'upstream'}}
        manifest['builds'] = {'benchmark': {'dockerfile':'Dockerfile', 'image':f'fixture-{name}:local', 'sources':['upstream']}}
        _write_json(source / 'extension.json', manifest)
        projects[name] = _local(source)
    root = tmp_path / 'host'
    _registry(root, projects)
    projected = registry_document(root)
    assert set(projected['builds']) == {'first/benchmark', 'second/benchmark'}
    assert set(projected['sources']) == {'first/self', 'first/upstream', 'second/self', 'second/upstream'}
    assert {row['provider_ref'] for row in projected['benchmarks'].values()} == {
        'plugin:benchmark:first/native', 'plugin:benchmark:second/native'}
    assert projected['builds']['first/benchmark']['dockerfile'] == 'external/extensions/first/self/Dockerfile'


def test_snapshot_callers_cannot_mutate_recorded_identity_or_locations(tmp_path):
    source = _project(tmp_path / 'source', category='module', identifier='fixture-module')
    catalog = _registry(tmp_path / 'host', {'fixture-module':_local(source)})
    snapshot = catalog.snapshot(config_refs=('module://fixture-module/sample.yaml',))
    expected = snapshot.sha256
    snapshot.lock['projects'].clear()
    snapshot.locations['projects'].clear()
    assert snapshot.sha256 == expected
    assert 'fixture-module' in snapshot.lock['projects']
    assert 'fixture-module' in snapshot.locations['projects']
