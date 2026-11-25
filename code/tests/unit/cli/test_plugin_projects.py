"""Standalone project acquisition, explicit configuration and frozen builds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.errors import ConfigCompatibilityError, ConfigSchemaError
from ovlab_benchctl.errors import ConfigReferenceError
from ovlab_benchctl.plugin_build import build_images
from ovlab_benchctl.plugin_projects import register_extension_project as register_project
from ovlab_benchctl.policy_plugins import plugin_for_type, registry_document, registry_scope
from ovlab_benchctl.strict_yaml import dumps, load

ROOT = Path(__file__).resolve().parents[4]


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def initialize(root):
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')


@pytest.fixture
def project(tmp_path, monkeypatch):
    root, plugin = tmp_path / 'ovlab', tmp_path / 'independent-plugin'
    shutil.copytree(ROOT / 'config', root / 'config')
    shutil.copytree(ROOT / 'examples/experiments', root / 'examples/experiments')
    (root / 'experiments/evaluation').mkdir(parents=True)
    for category in ('plugins', 'policies'):
        (root / 'configs' / category).mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'deploy/scripts', root / 'deploy/scripts')
    shutil.copytree(ROOT / 'examples/policy-plugin', plugin)
    (plugin / 'src/example_policy').rename(plugin / 'src/registered_policy')
    manifest = json.loads((plugin / 'extension.json').read_text())
    manifest['providers']['constant']['entrypoint'] = 'registered_policy.plugin:plugin'
    (plugin / 'extension.json').write_text(json.dumps(manifest))
    (root / 'configs/plugins').mkdir(exist_ok=True)
    (root / 'configs/extensions').mkdir()
    shutil.copytree(ROOT / 'examples/benchmark-plugin', root / 'examples/benchmark-plugin')
    (root / 'configs/extensions/registry.json').write_text(json.dumps({'schema_version': 'ovlab.extensions/v1',
        'projects': {'example-benchmark': {'source': {'kind':'local','path':'examples/benchmark-plugin'}}}}))
    (root / 'deploy/docker').mkdir()
    (root / 'deploy/docker/base-images.lock').write_text('fixture')
    (root / 'VERSION').write_text('0.2.0')
    shutil.copyfile(ROOT / '.gitignore', root / '.gitignore')
    monkeypatch.delenv('OVLAB_PLUGIN_REGISTRY', raising=False)
    import sys
    for name in list(sys.modules):
        if name == 'registered_policy' or name.startswith('registered_policy.'):
            monkeypatch.delitem(sys.modules, name)
    return root, plugin


def configure_experiment(root):
    override = {'schema_version': '0.1.0', 'kind': 'policy', 'type': 'example_policy',
                'extends': 'plugin://example-policy/policy.yaml',
                'settings': {'action': {'action_interface': 'plugin://example-policy/interfaces/planar-v1.yaml'}}}
    (root / 'configs/policies/custom.yaml').write_text(dumps(override))
    experiment = load(root / 'examples/benchmark-plugin/configs/evaluate.yaml')
    experiment['components']['policy'] = 'policies/custom.yaml'
    experiment['resources']['registry'] = 'plugin://example-policy/resources.yaml'
    (root / 'experiments/evaluation/custom.yaml').write_text(dumps(experiment))
    return 'experiments/evaluation/custom.yaml'


def test_local_registration_discovers_capabilities_and_configurations(project):
    root, source = project
    result = register_project(root, str(source))
    assert result['project'] == 'example-policy'
    registry = registry_document(root)
    assert registry['projects']['example-policy']['checkpoint_catalogs'] == ['resources.yaml']
    assert registry['plugins']['plugin:policy:example-policy/constant']['capabilities']['quantization'] == ['none']
    plugin = plugin_for_type('example_policy', root)
    assert plugin.implementation.api_version == '2'
    assert not (root / 'external').exists()
    assert register_project(root, str(source))['providers'] == ['constant']
    registration = json.loads((root / 'configs/extensions/registry.json').read_text())
    registration['projects']['example-policy']['source']['path'] = '../independent-plugin'
    (root / 'configs/extensions/registry.json').write_text(json.dumps(registration))
    assert register_project(root, str(source))['providers'] == ['constant']


def test_remote_registration_pins_manifest_and_adapter_without_manual_checkout(project):
    root, source = project
    initialize(source)
    pinned = git(source, 'rev-parse', 'HEAD')
    (source / 'README.md').write_text('later upstream change')
    git(source, 'commit', '-qam', 'later upstream')
    register_project(root, source.as_uri(), commit=pinned)
    registry = registry_document(root)
    assert registry['sources']['example-policy/self']['commit'] == pinned
    assert plugin_for_type('example_policy', root).implementation.api_version == '2'
    assert not (root / 'external').exists()
    assert (root / 'plugins/.cache').is_dir()
    from ovlab_benchctl.extensions import ExtensionCatalog
    cached = ExtensionCatalog(root).project_root('example-policy')
    (cached / 'README.md').write_text('unregistered cached edits')
    assert 'example-policy' not in registry_document(root)['projects']
    with pytest.raises(ConfigCompatibilityError, match='differs from its pinned commit'):
        plugin_for_type('example_policy', root, provider='plugin:policy:example-policy/constant')


def test_remote_dry_run_does_not_acquire_or_write_project(project):
    root, source = project
    before = (root / 'configs/extensions/registry.json').read_bytes()
    result = register_project(root, 'https://example.invalid/plugin.git', commit='a' * 40, dry_run=True)
    assert result['side_effects_performed'] is False
    assert (root / 'configs/extensions/registry.json').read_bytes() == before
    assert not (root / 'plugins').exists()
    with pytest.raises(ConfigSchemaError, match='Git commit'):
        register_project(root, 'https://example.invalid/plugin.git', commit='main')


def test_remote_plugin_cannot_acquire_undeclared_host_filesystem_dependencies(project):
    root, source = project
    manifest = json.loads((source / 'extension.json').read_text())
    manifest['sources']['host-files'] = {'kind': 'local', 'path': str(root / 'configs')}
    (source / 'extension.json').write_text(json.dumps(manifest))
    initialize(source)
    before = (root / 'configs/extensions/registry.json').read_bytes()
    with pytest.raises(ConfigCompatibilityError, match='local dependencies must stay inside'):
        register_project(root, source.as_uri(), commit=git(source, 'rev-parse', 'HEAD'))
    assert (root / 'configs/extensions/registry.json').read_bytes() == before


@pytest.mark.parametrize('options', [{'offline': True}, {'dry_run': True}])
def test_uncached_remote_manifest_is_not_fetched_for_offline_or_dry_run(project, options):
    root, source = project
    registration = {'schema_version': 'ovlab.extensions/v1', 'projects': {'unknown': {'source': {
        'kind': 'git', 'url': 'https://example.invalid/plugin.git', 'commit': 'a' * 40}}}}
    (root / 'configs/extensions/registry.json').write_text(json.dumps(registration))
    with pytest.raises(ConfigReferenceError, match='unknown or ambiguous'):
        build_images(root, ['unknown/policy'], **options)
    assert not (root / 'plugins').exists()


def test_invalid_manifest_does_not_change_registry(project):
    root, source = project
    before = (root / 'configs/extensions/registry.json').read_bytes()
    manifest = json.loads((source / 'extension.json').read_text())
    manifest['providers']['constant']['deployment']['image'] = 'unrelated:image'
    (source / 'extension.json').write_text(json.dumps(manifest))
    with pytest.raises(ConfigSchemaError, match='image differs'):
        register_project(root, str(source))
    assert (root / 'configs/extensions/registry.json').read_bytes() == before


def test_build_bakes_explicit_user_extensions_and_selected_plugin(project):
    root, source = project
    register_project(root, str(source))
    experiment = configure_experiment(root)
    initialize(root)
    archives = []

    def docker(command, **kwargs):
        context = Path(command[-1])
        assert (context / 'external/extensions/example-policy/self/src/registered_policy/plugin.py').is_file()
        for index, value in enumerate(command):
            if value != '--secret':
                continue
            declaration = command[index + 1]
            if declaration.startswith('id=ovlab_config_bundle,'):
                archive_path = Path(declaration.split('src=', 1)[1])
                archives.append(archive_path)
                with tarfile.open(archive_path) as archive:
                    assert archive.extractfile('policies/custom.yaml').read().decode().find('plugin://example-policy/policy.yaml') >= 0
                    assert archive.extractfile('plugins/configs/example-policy/policy.yaml')
                    assert archive.extractfile('plugins/projects/example-policy/src/registered_policy/plugin.py')
        return SimpleNamespace(returncode=0)

    plans = build_images(root, config=experiment, runner=docker)
    assert {plan['role'] for plan in plans} == {'example-policy/policy-example', 'example-benchmark/benchmark-example'}
    assert len(plans[0]['config_bundle_sha256']) == 64
    assert 'OVLAB_CONFIG_BUNDLE_SHA256=' + plans[0]['config_bundle_sha256'] in plans[0]['command']
    assert archives and all(not path.exists() for path in archives)
    assert not (root / 'external').exists()
    changed = load(root / 'configs/policies/custom.yaml')
    changed['settings']['checkpoint_id'] = 'another-explicit-checkpoint'
    (root / 'configs/policies/custom.yaml').write_text(dumps(changed))
    preview = build_images(root, config=experiment, dry_run=True)
    assert preview[0]['config_bundle_sha256'] != plans[0]['config_bundle_sha256']


def test_plugin_can_declare_quantization_profiles_without_changing_core(project):
    from ovlab_benchctl.model_deployment import ModelDeployment
    root, source = project
    manifest = json.loads((source / 'extension.json').read_text())
    manifest['providers']['constant']['deployment']['quantization'].append('vendor-int3')
    (source / 'extension.json').write_text(json.dumps(manifest))
    implementation = source / 'src/registered_policy/plugin.py'
    implementation.write_text(implementation.read_text().replace(
        "runtime['quantization'] != 'none'", "runtime['quantization'] not in {'none', 'vendor-int3'}"))
    register_project(root, str(source))
    plan = ModelDeployment(root, environment={}).plan('plugin://example-policy/deployment.yaml', quantization='vendor-int3')
    assert plan.quantization == 'vendor-int3'


def test_bundle_refuses_changed_project_code_before_materializing(project):
    root, source = project
    register_project(root, str(source))
    experiment = configure_experiment(root)
    builder = ConfigBundleBuilder(root)
    bundle = builder.build(experiment, None)
    (source / 'src/registered_policy/runtime.py').write_text('# edited after planning\n')
    with pytest.raises(ValueError, match='changed after bundling'):
        with builder.materialize(bundle):
            pytest.fail('changed code cannot retain the old bundle identity')


def test_flat_layout_plugin_control_code_is_available_in_frozen_worker(project, tmp_path):
    root, source = project
    (source / 'src/registered_policy').rename(source / 'registered_policy')
    (source / 'src').rmdir()
    pyproject = source / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text().replace('where = ["src"]', 'where = ["."]'))
    register_project(root, str(source))
    builder = ConfigBundleBuilder(root)
    bundle = builder.build(configure_experiment(root), None)
    with builder.materialize(bundle) as config:
        worker = tmp_path / 'flat-worker'
        shutil.copytree(config, worker / 'configs')
        env = dict(os.environ, OVLAB_ROOT=str(worker),
                   OVLAB_PLUGIN_REGISTRY=str(worker / 'configs/plugins/registry.json'),
                   PYTHONPATH=os.pathsep.join(str(path) for path in ROOT.glob('code/**/src')))
        script = (
            'from pathlib import Path; from ovlab_benchctl.resolver import ConfigResolver; '
            f'r=ConfigResolver(Path({str(worker / "configs")!r}), repository_root=Path({str(worker)!r})); '
            "r.load_component('plugin://example-policy/policy.yaml','policy'); "
            'import registered_policy.plugin; '
            f'assert Path(registered_policy.plugin.__file__).is_relative_to(Path({str(worker)!r}))'
        )
        result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_project_check_rejects_missing_declared_configuration_template(project):
    from ovlab_benchctl.plugin_catalog import check_project
    root, source = project
    manifest_path = source / 'extension.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault('configs', {})['templates'] = {'evaluate': ['missing.yaml']}
    manifest_path.write_text(json.dumps(manifest))
    register_project(root, str(source))
    with pytest.raises(ConfigReferenceError, match='missing its configuration template'):
        check_project(root, 'example-policy')


def test_plugin_cli_selection_uses_project_identity_not_variant_or_policy_type(project, monkeypatch):
    from ovlab_benchctl import cli
    from ovlab_benchctl.plugin_catalog import check_project, list_projects
    root, source = project
    manifest_path = source / 'extension.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['providers']['constant']['id'] = 'constant-variant'
    manifest_path.write_text(json.dumps(manifest))
    register_project(root, str(source))
    monkeypatch.setenv('OVLAB_ROOT', str(root))
    assert 'example-policy' in {row['id'] for row in list_projects(root)}
    assert check_project(root, 'example-policy')['model_loaded'] is False
    for invalid in ('constant-variant', 'example_policy'):
        with pytest.raises(ConfigReferenceError, match='unknown extension project'):
            cli._dispatch(cli._parser().parse_args(['plugin', 'inspect', invalid]))
