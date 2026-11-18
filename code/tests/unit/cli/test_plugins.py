"""Prove third-party sources and adapters work without editing OVLAB dispatch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.deployment import ComposeDeployment
from ovlab_benchctl.errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from ovlab_benchctl.model_deployment import ModelDeployment
from ovlab_benchctl.plugin_build import build_images
from ovlab_benchctl.plugin_sources import sync_sources, tree_identity
from ovlab_benchctl.policy_plugins import plugin_for_type, registry_document
from ovlab_benchctl.resolver import ConfigResolver
from ovlab_benchctl.strict_yaml import dumps, load

ROOT = Path(__file__).resolve().parents[4]


def write_registry(root, document):
    path = root / 'configs/plugins/registry.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def test_remote_clone_uses_commit_not_moving_branch_and_protects_local_edits(tmp_path):
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    git(upstream, 'config', 'user.name', 'Fixture')
    git(upstream, 'config', 'user.email', 'fixture@example.invalid')
    (upstream / 'model.py').write_text('VERSION = 1\n')
    git(upstream, 'add', '.')
    git(upstream, 'commit', '-qm', 'first')
    pinned = git(upstream, 'rev-parse', 'HEAD')
    (upstream / 'model.py').write_text('VERSION = 2\n')
    git(upstream, 'commit', '-qam', 'second')
    root = tmp_path / 'ovlab'
    registry = {'schema_version': 'ovlab.plugins/v1', 'sources': {'toy': {'kind': 'git', 'url': str(upstream), 'commit': pinned}}, 'plugins': {}, 'builds': {}}
    write_registry(root, registry)
    sync_sources(root, dry_run=True)
    assert not (root / 'external').exists()
    with pytest.raises(ConfigReferenceError, match='offline'):
        sync_sources(root, offline=True)
    rows = sync_sources(root)
    destination = root / 'external/toy'
    assert rows[0]['revision'] == pinned
    assert (destination / 'model.py').read_text() == 'VERSION = 1\n'
    assert sync_sources(root, offline=True)[0]['content_sha256'] == rows[0]['content_sha256']
    (destination / 'model.py').write_text('local work\n')
    with pytest.raises(ConfigCompatibilityError, match='local changes'):
        sync_sources(root)
    assert (destination / 'model.py').read_text() == 'local work\n'
    registry['sources']['toy']['commit'] = 'main'
    write_registry(root, registry)
    with pytest.raises(ConfigSchemaError, match='full lowercase Git commit'):
        registry_document(root)


def test_local_snapshot_hashes_bytes_and_rejects_snapshot_edits_and_escape(tmp_path):
    root, original = tmp_path / 'ovlab', tmp_path / 'project'
    original.mkdir()
    (original / 'model.py').write_text('first')
    write_registry(root, {'schema_version': 'ovlab.plugins/v1', 'sources': {'toy': {'kind': 'local', 'path': str(original)}}, 'plugins': {}, 'builds': {}})
    first = sync_sources(root)[0]
    (original / 'model.py').write_text('second')
    second = sync_sources(root)[0]
    assert first['content_sha256'] != second['content_sha256']
    snapshot = root / 'external/toy/model.py'
    assert snapshot.read_text() == 'second'
    snapshot.write_text('snapshot edits')
    with pytest.raises(ConfigCompatibilityError, match='snapshot has changes'):
        sync_sources(root)
    assert snapshot.read_text() == 'snapshot edits'
    (original / 'outside').symlink_to(tmp_path)
    with pytest.raises(ConfigReferenceError, match='symlink escapes'):
        tree_identity(original)


@pytest.fixture
def external_plugin(tmp_path, monkeypatch):
    for name in list(sys.modules):
        if name == "example_policy" or name.startswith("example_policy."):
            monkeypatch.delitem(sys.modules, name)
    root = tmp_path / 'ovlab'
    shutil.copytree(ROOT / 'configs', root / 'configs')
    shutil.copytree(ROOT / 'experiments/evaluation', root / 'experiments/evaluation')
    shutil.copytree(ROOT / 'examples/policy-plugin', root / 'examples/policy-plugin')
    shutil.copytree(ROOT / 'deploy/compose', root / 'deploy/compose')
    shutil.copytree(ROOT / 'deploy/scripts', root / 'deploy/scripts')
    (root / 'deploy/docker').mkdir()
    (root / 'deploy/docker/base-images.lock').write_text('fixture base lock')
    (root / 'VERSION').write_text('0.2.0')
    registry = json.loads((ROOT / 'examples/policy-plugin/registry.json').read_text())
    write_registry(root, registry)
    monkeypatch.delenv('OVLAB_PLUGIN_REGISTRY', raising=False)
    policy = {'schema_version': '0.1.0', 'kind': 'policy', 'type': 'example_policy', 'settings': {
        'checkpoint_id': 'openvla-7b', 'input': {}, 'runtime': {'quantization': 'none'},
        'action': {'action_interface': 'interfaces/actions/mock-delta-gripper-v1.yaml'}, 'raw_output': {'enabled': False}}}
    path = root / 'configs/policies/example.yaml'
    path.write_text(dumps(policy))
    experiment = load(root / 'experiments/evaluation/mock-e2e-smoke.yaml')
    experiment['components']['policy'] = 'policies/example.yaml'
    experiment['components']['protocol'] = 'protocols/example.yaml'
    (root / 'experiments/evaluation/example-plugin.yaml').write_text(dumps(experiment))
    protocol = load(root / 'configs/protocols/smoke-v1.yaml')
    protocol['recording']['raw_policy_output'] = False
    (root / 'configs/protocols/example.yaml').write_text(dumps(protocol))
    profile = load(root / 'configs/local/profile.example.yaml')
    profile['paths'] = {key: str(tmp_path / key) for key in profile['paths']}
    profile['resources'] = {'checkpoints': {}}
    (root / 'configs/local/example.yaml').write_text(dumps(profile))
    deploy = {'schema_version': 'ovlab.deployment/v1', 'kind': 'deployment', 'id': 'example',
              'components': {'policy': 'policies/example.yaml'}, 'resources': {'registry': 'resources/registry.yaml'}, 'build': True}
    (root / 'configs/deploy/example.yaml').write_text(dumps(deploy))
    return root


def test_external_plugin_resolves_predicts_and_plans_deployment_and_evaluation(external_plugin):
    root = external_plugin
    resolver = ConfigResolver(root / 'configs', repository_root=root)
    resolved = resolver.resolve('experiments/evaluation/example-plugin.yaml', local_profile='configs/local/example.yaml', environment={})
    plugin = plugin_for_type('example_policy', root)
    adapter = plugin.implementation.create_adapter(resolved.policy_settings)
    from helpers.contexts import make_run_context, make_episode_context
    from helpers.mock_benchmark import MockBenchmark
    benchmark = MockBenchmark()
    run, episode = make_run_context(), make_episode_context()
    benchmark.initialize(run)
    observation = benchmark.reset_episode(episode).initial_observation
    caps = adapter.initialize(run)
    adapter.reset_episode(episode)
    prediction = adapter.predict(observation)
    assert prediction.actions.shape == (1, 3)
    assert (prediction.actions == 0).all()
    assert plugin.implementation.identity(caps)['method_identity']['id'] == 'example-constant'
    adapter.end_episode(episode)
    adapter.close()
    deployment = ModelDeployment(root, environment={})
    plan = deployment.plan('example')
    preview = deployment.run(plan, dry_run=True)
    assert preview['role'] == 'policy-example'
    assert preview['build_command'][-1] == 'policy-example'
    assert '/opt/ovlab/configs/plugins/registry.json' in ' '.join(preview['run_command'])
    bundle_builder = ConfigBundleBuilder(root)
    bundle_before_sync = bundle_builder.build('configs/deploy/example.yaml', None, kind='deployment')
    sync_sources(root)
    bundle = bundle_builder.build('configs/deploy/example.yaml', None, kind='deployment')
    assert bundle.sha256 == bundle_before_sync.sha256
    with bundle_builder.materialize(bundle) as config:
        document = json.loads((config / 'plugins/registry.json').read_text())
        assert document['plugins']['example_policy']['entrypoint'] == 'example_policy.plugin:plugin'
        assert len(document['source_identities']['example-policy']['content_sha256']) == 64
    assert not config.exists()
    evaluation = load(root / 'experiments/evaluation/libero10-vanilla-rpc-smoke.yaml')
    evaluation['components']['policy'] = 'policies/example.yaml'
    evaluation['deployment']['profile'] = 'example'
    (root / 'experiments/evaluation/example-compose.yaml').write_text(dumps(evaluation))
    compose = ComposeDeployment(root, environment={})
    evaluation_plan = compose.plan('example-compose', env_file='deploy/compose/.env.example')
    spec = compose.service_spec(evaluation_plan)
    assert set(spec['services']) == {'policy-example', 'benchmark-example'}
    with compose._compose_spec(evaluation_plan, []):
        assert Path(evaluation_plan.compose_files[-1]).is_file()
    assert not Path(evaluation_plan.compose_files[-1]).exists()
    assert compose.run(evaluation_plan, dry_run=True)['plugin_services'] == spec
    (root / 'examples/policy-plugin/src/example_policy/runtime.py').write_text('# changed local runtime')
    assert bundle_builder.build('configs/deploy/example.yaml', None, kind='deployment').sha256 != bundle.sha256


def test_registry_build_dispatch_handles_new_role_and_cleans_secret_files(external_plugin):
    root = external_plugin
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    calls, secrets = [], []
    def runner(command, **kwargs):
        calls.append(command)
        for i, value in enumerate(command):
            if value == '--secret':
                path = Path(command[i + 1].split('src=', 1)[1])
                assert path.is_file()
                secrets.append(path)
        return SimpleNamespace(returncode=0)
    plans = build_images(root, ['policy-example'], runner=runner)
    assert len(calls) == 1
    assert plans[0]['image'] == 'ovlab-policy-example:local'
    assert str(root / 'external/example-policy/Dockerfile') in calls[0]
    assert any('plugin-registry.sha256=' in value for value in calls[0])
    assert f"OVLAB_VERSION={(root / 'VERSION').read_text().strip()}" in calls[0]
    assert all(not path.exists() for path in secrets)


def test_service_startup_selects_external_plugin_without_builtin_settings_dispatch(external_plugin, monkeypatch):
    from ovlab_benchctl.application import OvlabApplication
    import ovlab_remote_policy.service
    captured = {}

    class Service:
        def __init__(self, socket, adapter, *, identity_provider, startup_context, **kwargs):
            self.adapter, self.identity_provider, self.startup_context = adapter, identity_provider, startup_context

        def serve(self):
            capabilities = self.adapter.initialize(self.startup_context)
            captured.update(self.identity_provider(capabilities))
            self.adapter.close()

    monkeypatch.setattr(ovlab_remote_policy.service, 'PolicyService', Service)
    root = external_plugin
    app = OvlabApplication(root, environment={'OVLAB_LOCAL_PROFILE': str(root / 'configs/local/example.yaml')})
    result = app.serve('configs/deploy/example.yaml', host='127.0.0.1', port=8123)
    assert result['closed'] is True
    assert captured['method_identity']['id'] == 'example-constant'


def test_listing_and_policy_deploy_planning_need_only_standard_library(external_plugin):
    env = dict(os.environ, OVLAB_ROOT=str(external_plugin))
    # Explicitly remove site-packages and test helper paths.
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in (ROOT / 'code').glob('**/src'))
    completed = subprocess.run([sys.executable, '-S', '-m', 'ovlab_benchctl.cli', 'deploy', 'run', 'example', '--dry-run', '--json'], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)['result']['role'] == 'policy-example'


def test_existing_image_is_rejected_before_checkpoint_when_plugin_source_changed(external_plugin, monkeypatch):
    from dataclasses import replace
    from ovlab_benchctl.plugin_identity import PLUGIN_BUILD_LABEL
    from ovlab_benchctl.image_metadata import IMAGE_CONTRACT_LABEL
    root = external_plugin
    calls = []

    def docker(command, **kwargs):
        calls.append(command)
        if command[:3] == ['docker', 'container', 'inspect']:
            return SimpleNamespace(returncode=1, stdout='', stderr='No such container')
        return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config': {'Labels': {
            IMAGE_CONTRACT_LABEL: 'resolved-checkpoint-quantization-config-bundle-v2',
            'io.github.ovlab.service.transport': 'tcp-persistent-v1',
            'io.github.ovlab.role': 'policy-example', PLUGIN_BUILD_LABEL: '0'*64}}}]), stderr='')

    owner = ModelDeployment(root, environment={'OVLAB_MODEL_DATA_ROOT': str(root/'data')}, runner=docker)
    monkeypatch.setattr(owner, '_checkpoint', lambda plan: pytest.fail('stale image must fail before checkpoint resolution'))
    with pytest.raises(ConfigCompatibilityError, match='does not match registered plugin'):
        owner.run(replace(owner.plan('example'), build=False))
    assert not any(command[:2] == ['docker', 'run'] for command in calls)


def test_plugin_image_identity_tracks_local_bytes_but_not_unrelated_registrations(external_plugin):
    from ovlab_benchctl.plugin_identity import plugin_build_identity
    root = external_plugin
    first = plugin_build_identity(root, 'policy-example')
    registry = registry_document(root)
    registry['plugins']['other'] = dict(id='other', family='other', runtime_status='unavailable', entrypoint=None, sources=[])
    write_registry(root, registry)
    assert plugin_build_identity(root, 'policy-example') == first
    (root/'examples/policy-plugin/src/example_policy/runtime.py').write_text('# edited implementation')
    assert plugin_build_identity(root, 'policy-example') != first


@pytest.mark.parametrize('mutation, message', [
    ('omitted-source', 'omits required plugin sources'),
    ('different-image', 'image differs'),
    ('reserved-service', 'reserved infrastructure service'),
    ('recipe-escape', 'paths must stay below'),
    ('duplicate-document', 'registered by multiple plugins'),
])
def test_invalid_registrations_fail_before_import_or_fetch(external_plugin, mutation, message):
    from copy import deepcopy
    root = external_plugin
    registry = registry_document(root)
    plugin = registry['plugins']['example_policy']
    if mutation == 'omitted-source': registry['builds']['policy-example']['sources'] = []
    if mutation == 'different-image': plugin['deployment']['image'] = 'different:local'
    if mutation == 'reserved-service': plugin['deployment']['policy_service'] = 'reporting'
    if mutation == 'recipe-escape': registry['builds']['policy-example']['dockerfile'] = '../outside/Dockerfile'
    if mutation == 'duplicate-document':
        plugin['document_kinds'] = ['custom_descriptor']
        registry['plugins']['second'] = deepcopy(plugin)
        registry['plugins']['second'].update(id='second', config_type='second')
    write_registry(root, registry)
    with pytest.raises(ConfigSchemaError, match=message):
        registry_document(root)
    assert not (root/'external').exists()


def test_explicit_python_environment_selects_registry_without_process_mutation(external_plugin, monkeypatch):
    from ovlab_benchctl.application import OvlabApplication
    root = external_plugin
    registry = registry_document(root)
    registry['builds']['policy-example']['image'] = 'example-override:local'
    registry['plugins']['example_policy']['deployment']['image'] = 'example-override:local'
    override = root/'override.json'
    override.write_text(json.dumps(registry))
    monkeypatch.setenv('OVLAB_PLUGIN_REGISTRY', '/nonexistent-global-registry.json')
    owner = ModelDeployment(root, environment={'OVLAB_PLUGIN_REGISTRY': str(override)})
    assert owner.plan('example').image == 'example-override:local'
    app = OvlabApplication(root, environment={'OVLAB_PLUGIN_REGISTRY': str(override)})
    assert app.resolve('configs/policies/example.yaml')['type'] == 'example_policy'
    assert os.environ['OVLAB_PLUGIN_REGISTRY'] == '/nonexistent-global-registry.json'


def test_local_source_replacement_restores_previous_snapshot_on_rename_failure(tmp_path, monkeypatch):
    root, original = tmp_path/'ovlab', tmp_path/'project'
    original.mkdir()
    (original/'runtime.py').write_text('first')
    write_registry(root, {'schema_version': 'ovlab.plugins/v1', 'sources': {'toy': {'kind': 'local', 'path': str(original)}}, 'plugins': {}, 'builds': {}})
    sync_sources(root)
    (original/'runtime.py').write_text('second')
    destination = root/'external/toy'
    rename = Path.rename

    def fail_publish(source, target):
        if source.name.startswith('.toy-') and '-previous-' not in source.name and Path(target) == destination:
            raise OSError('simulated publish failure')
        return rename(source, target)

    monkeypatch.setattr(Path, 'rename', fail_publish)
    with pytest.raises(OSError, match='publish failure'):
        sync_sources(root)
    assert (destination/'runtime.py').read_text() == 'first'
    assert not list((root/'external').glob('.toy-*'))


def test_source_sync_refuses_symlinked_workspace_without_writing_outside(tmp_path):
    root, original, outside = tmp_path/'ovlab', tmp_path/'project', tmp_path/'outside'
    original.mkdir(); outside.mkdir()
    (original/'runtime.py').write_text('source')
    write_registry(root, {'schema_version': 'ovlab.plugins/v1', 'sources': {'toy': {'kind': 'local', 'path': str(original)}}, 'plugins': {}, 'builds': {}})
    (root/'external').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigReferenceError, match='cannot be symlinks'):
        sync_sources(root)
    assert list(outside.iterdir()) == []


def test_build_detects_mutated_staged_source_and_cleans_secrets(external_plugin):
    root = external_plugin
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    secrets = []

    def docker(command, **kwargs):
        for index, value in enumerate(command):
            if value == '--secret': secrets.append(Path(command[index+1].split('src=',1)[1]))
        (root/'external/example-policy/src/example_policy/runtime.py').write_text('# changed during build')
        return SimpleNamespace(returncode=0)

    with pytest.raises(ConfigCompatibilityError, match='changed during build'):
        build_images(root, ['policy-example'], runner=docker)
    assert all(not path.exists() for path in secrets)


@pytest.mark.parametrize('mount', [
    {'source': '/tmp/data', 'target': '/opt/ovlab'},
    {'source': 'relative', 'target': '/models'},
    {'source': '/tmp/data', 'target': '/models/../tmp'},
    {'source': '/tmp/data', 'target': '/models', 'required_files': ['../private']},
    {'source': '/tmp/data', 'target': '/models', 'environment': 'OVLAB_PLUGIN_REGISTRY'},
])
def test_plugin_mount_contract_rejects_invalid_or_overlapping_paths(mount):
    from ovlab_benchctl.plugin_mounts import validate_runtime_mounts
    with pytest.raises(ConfigCompatibilityError):
        validate_runtime_mounts([mount])


def test_persistent_deployment_checks_required_mount_files_before_build(external_plugin, monkeypatch):
    from ovlab_policy_sdk.plugin import PolicyPlugin
    root = external_plugin
    directory = root/'adapter'
    directory.mkdir()
    monkeypatch.setattr(PolicyPlugin, 'runtime_mounts', lambda *args: [
        {'source': str(directory), 'target': '/adapters/extra', 'required_files': ['required.json']}])
    calls = []
    owner = ModelDeployment(root, environment={'OVLAB_MODEL_DATA_ROOT': str(root/'data')},
                            runner=lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=1, stdout='', stderr='No such container')))
    with pytest.raises(ConfigCompatibilityError, match='required.json'):
        owner.run(owner.plan('example'))
    assert not any(command[0] == 'bash' for command in calls)


def test_documented_cpu_example_configures_and_verifies_its_tiny_artifact(external_plugin):
    import runpy
    from ovlab_benchctl.checkpointing import CheckpointResolver, checkpoint_spec
    from ovlab_benchctl.application import OvlabApplication
    from ovlab_benchctl.model_deployment import resolve_policy_service
    root = external_plugin
    configure = runpy.run_path(str(root/'examples/policy-plugin/configure.py'))['configure']
    paths = configure(root, dry_run=True)
    assert all(not (root/path).exists() for path in paths)
    configure(root)
    assert configure(root) == paths
    spec = checkpoint_spec(root, 'configs/deploy/example-plugin.yaml')
    resolver = CheckpointResolver(global_cache=root/'empty-cache', managed_cache=root/'cache')
    resolved = resolver.resolve(spec, local_path=root/'examples/policy-plugin/fixtures/checkpoint', offline=True)
    assert resolved.verified_file_count == 1
    owner = ModelDeployment(root, environment={})
    preview = owner.run(owner.plan('example-plugin', offline=True), dry_run=True)
    assert '--gpus' not in preview['run_command']
    app = OvlabApplication(root, environment={'OVLAB_LOCAL_PROFILE': str(root/'configs/local/example-plugin.yaml')})
    settings = resolve_policy_service(app, root/'configs/deploy/example-plugin.yaml').policy_settings
    assert settings.checkpoint['local_path'].endswith('fixtures/checkpoint')
    edited = root/paths[0]
    edited.write_text(edited.read_text()+'\n# developer changes\n')
    with pytest.raises(ValueError, match='preserving modified configuration'):
        configure(root)


def test_remote_sources_reject_ignored_code_but_allow_excluded_python_caches(tmp_path):
    from ovlab_benchctl.plugin_sources import inspect_sources
    project = tmp_path/'upstream'
    project.mkdir()
    git(project, 'init', '-q')
    git(project, 'config', 'user.name', 'Fixture')
    git(project, 'config', 'user.email', 'fixture@example.invalid')
    (project/'.gitignore').write_text('ignored.py\n')
    (project/'runtime.py').write_text('VERSION = 1\n')
    git(project, 'add', '.')
    git(project, 'commit', '-qm', 'fixture')
    root = tmp_path/'ovlab'
    write_registry(root, {'schema_version':'ovlab.plugins/v1', 'sources':{'toy':{'kind':'git','url':str(project),'commit':git(project,'rev-parse','HEAD')}},'plugins':{},'builds':{}})
    sync_sources(root)
    snapshot = root/'external/toy'
    (snapshot/'__pycache__').mkdir()
    (snapshot/'__pycache__/runtime.pyc').write_bytes(b'cache')
    assert sync_sources(root, offline=True)[0]['dirty'] is False
    (snapshot/'ignored.py').write_text('UNPINNED = True')
    with pytest.raises(ConfigCompatibilityError, match='local changes'):
        sync_sources(root, offline=True)
    assert inspect_sources(root)[0]['dirty'] is True


def test_import_cache_cannot_silently_reuse_a_different_local_checkout(external_plugin):
    root = external_plugin
    original = plugin_for_type('example_policy', root)
    assert original.implementation is original.implementation
    second = root/'other-project'
    shutil.copytree(root/'examples/policy-plugin', second)
    registry = registry_document(root)
    registry['sources']['example-policy']['path'] = str(second)
    write_registry(root, registry)
    with pytest.raises(ConfigCompatibilityError, match='different source; restart'):
        plugin_for_type('example_policy', root).implementation


@pytest.mark.parametrize('mutation, message', [('escape', 'relative checkpoint file'), ('digest', 'aggregate SHA-256')])
def test_checkpoint_manifest_validation_rejects_path_escape_and_mismatched_aggregate(external_plugin, mutation, message):
    from ovlab_benchctl.schema import validate
    registry = load(external_plugin/'examples/policy-plugin/configs/resources.yaml')
    entry = registry['checkpoints']['example-constant']
    if mutation == 'escape': entry['files']['../metadata.json'] = entry['files'].pop('metadata.json')
    else: entry['expected_sha256'] = '0'*64
    with pytest.raises(ConfigSchemaError, match=message):
        validate(registry, 'example', 'resource_registry', repository_root=external_plugin)


def test_settings_without_required_recording_contract_have_clear_plugin_error(external_plugin, monkeypatch):
    root = external_plugin
    plugin = plugin_for_type('example_policy', root).implementation
    monkeypatch.setattr(plugin, 'resolve_settings', lambda *args, **kwargs: SimpleNamespace())
    resolver = ConfigResolver(root/'configs', repository_root=root)
    with pytest.raises(ConfigCompatibilityError, match='boolean record_raw_output'):
        resolver.resolve('experiments/evaluation/example-plugin.yaml', local_profile='configs/local/example.yaml', environment={})
