"""Shared registration invariants across plugin and module projects."""
import json
from pathlib import Path
import subprocess

import pytest

from ovlab_benchctl.errors import ConfigCompatibilityError, ConfigSchemaError
from ovlab_benchctl.extensions import ExtensionCatalog, ProviderRef, acquire_manifest
from ovlab_benchctl.plugin_projects import register_extension_project


def project(root, name='sample', category='module'):
    path = root / name
    (path / 'configs').mkdir(parents=True)
    (path / 'configs/default.yaml').write_text('kind: example\nvalue: 1\n')
    manifest = {'schema_version': 'ovlab.extension/v1', 'id': name, 'category': category,
                'version': '1.0.0', 'providers': {}, 'configs': {'root': 'configs', 'templates': {'example': ['default.yaml']}}}
    (path / 'extension.json').write_text(json.dumps(manifest))
    return path


def document(**projects):
    return {'schema_version': 'ovlab.extensions/v1', 'projects': {
        name: {'source': {'kind': 'local', 'path': str(path)}} for name, path in projects.items()}}


def test_shared_registration_is_idempotent_and_category_checked(tmp_path):
    root = tmp_path / 'root'
    (root / 'plugins').mkdir(parents=True)
    (root / 'plugins/registry.json').write_text(json.dumps(document()))
    source = project(tmp_path)
    assert register_extension_project(root, str(source), category='module')['project'] == 'sample'
    target = root / 'configs/extensions/registry.json'
    before = target.read_bytes()
    assert register_extension_project(root, str(source), category='module')['project'] == 'sample'
    assert target.read_bytes() == before
    with pytest.raises(ConfigCompatibilityError, match='expected a plugin'):
        register_extension_project(root, str(source), category='plugin')


def test_registration_dry_run_leaves_no_registry(tmp_path):
    root = tmp_path / 'root'
    (root / 'plugins').mkdir(parents=True)
    (root / 'plugins/registry.json').write_text(json.dumps(document()))
    result = register_extension_project(root, str(project(tmp_path)), category='module', dry_run=True)
    assert result['side_effects_performed'] is False
    assert not (root / 'configs').exists()


def test_local_source_names_are_project_scoped(tmp_path):
    one, two = project(tmp_path, 'one'), project(tmp_path, 'two')
    for path in (one, two):
        (path / 'upstream').mkdir()
        manifest = json.loads((path / 'extension.json').read_text())
        manifest['sources'] = {'runtime': {'kind': 'local', 'path': 'upstream'}}
        manifest['builds'] = {'worker': {'image': 'example:test', 'dockerfile': 'Dockerfile', 'sources': ['runtime']}}
        (path / 'extension.json').write_text(json.dumps(manifest))
    catalog = ExtensionCatalog(tmp_path, document=document(one=one, two=two))
    assert set(catalog.project_sources('one')) == {'one/self', 'one/runtime'}
    assert set(catalog.project_sources('two')) == {'two/self', 'two/runtime'}
    assert set(catalog.project_builds('one')) == {'one/worker'}


def test_git_cache_verifies_origin_and_preserves_unregistered_edits(tmp_path):
    source = project(tmp_path)
    def git(path, *args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Fixture')
    git(source, 'config', 'user.email', 'fixture@example.invalid')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'fixture')
    binding = {'kind': 'git', 'url': source.as_uri(), 'commit': git(source, 'rev-parse', 'HEAD')}
    root, manifest = acquire_manifest(tmp_path / 'root', binding)
    assert manifest['id'] == 'sample'
    catalog = ExtensionCatalog(tmp_path / 'root', document={'schema_version': 'ovlab.extensions/v1',
                              'projects': {'sample': {'source': binding}}})
    assert catalog.project_root('sample') == root
    git(root, 'remote', 'set-url', 'origin', 'https://example.invalid/other.git')
    with pytest.raises(ConfigCompatibilityError, match='origin'):
        catalog.project_manifest('sample')
    git(root, 'remote', 'set-url', 'origin', source.as_uri())
    (root / 'edited.txt').write_text('preserve me')
    with pytest.raises(ConfigCompatibilityError, match='pinned commit'):
        catalog.project_manifest('sample')
    assert (root / 'edited.txt').read_text() == 'preserve me'


@pytest.mark.parametrize('value', ['plugin:metric:a/b', 'module:benchmark:a/b', 'plugin:policy:a/../b'])
def test_invalid_typed_bindings_are_rejected(value):
    with pytest.raises(ConfigSchemaError):
        ProviderRef.parse(value)


def test_frozen_catalog_uses_only_locked_locations_and_rejects_missing_code(tmp_path):
    import shutil
    from ovlab_benchctl.errors import ConfigReferenceError
    source = project(tmp_path / 'host')
    registration = document(sample=source)
    catalog = ExtensionCatalog(tmp_path / 'host', document=registration)
    snapshot = catalog.snapshot(project_ids=['sample'])
    worker = tmp_path / 'worker'
    configs = worker / 'configs'
    bundled = configs / 'modules/projects/sample'
    shutil.copytree(source, bundled)
    shutil.copytree(source / 'configs', configs / 'modules/configs/sample')
    (configs / 'extensions').mkdir()
    (configs / '.ovlab-bundle.json').write_text('{}')
    (configs / 'extensions/registry.json').write_text(json.dumps(registration))
    (configs / 'extensions/lock.json').write_text(json.dumps({'lock': snapshot.lock, 'locations': {
        'projects': {'sample': 'configs/modules/projects/sample'},
        'sources': {'sample/self': 'configs/modules/projects/sample'}}}))
    frozen = ExtensionCatalog(worker, registry_path=configs / 'extensions/registry.json')
    assert frozen.project_root('sample') == bundled
    assert frozen.config_path('module://sample/default.yaml') == configs / 'modules/configs/sample/default.yaml'
    shutil.rmtree(bundled)
    assert source.exists()
    with pytest.raises(ConfigReferenceError, match='frozen.*missing'):
        frozen.project_root('sample')


def test_build_registry_identity_ignores_unselected_registration_and_host_location(tmp_path):
    import shutil
    root = tmp_path / 'one'
    (root / 'plugins').mkdir(parents=True)
    source = project(root / 'plugins', 'sample', 'plugin')
    manifest = json.loads((source / 'extension.json').read_text())
    manifest['builds'] = {'worker': {'image': 'example:test', 'dockerfile': 'Dockerfile', 'sources': []}}
    (source / 'extension.json').write_text(json.dumps(manifest))
    (source / 'Dockerfile').write_text('FROM scratch\n')
    # The legacy projection only handles policy providers; use the neutral selected lock.
    first = ExtensionCatalog(root, document=document(sample=source)).snapshot(build_refs=['sample/worker'])
    other = tmp_path / 'two'
    shutil.copytree(root, other)
    after = document(sample=other / 'plugins/sample')
    after['projects']['unavailable'] = {'source': {'kind': 'git', 'url': 'https://example.invalid/missing.git', 'commit': 'a' * 40}}
    second = ExtensionCatalog(other, document=after).snapshot(build_refs=['sample/worker'])
    assert first.sha256 == second.sha256
    assert str(root) not in json.dumps(first.lock)
