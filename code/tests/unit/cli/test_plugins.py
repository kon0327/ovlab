"""Prove third-party sources and adapters work without editing OVLAB dispatch."""
import json
import os
from pathlib import Path
import runpy
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
    shutil.copytree(ROOT / 'config', root / 'config')
    shutil.copytree(ROOT / 'examples/experiments', root / 'examples/experiments')
    (root / 'experiments/evaluation').mkdir(parents=True)
    for category in ('resources', 'policies', 'protocols', 'local', 'deploy'):
        (root / 'configs' / category).mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'examples/policy-plugin', root / 'examples/policy-plugin')
    shutil.copytree(ROOT / 'deploy/compose', root / 'deploy/compose')
    shutil.copytree(ROOT / 'deploy/scripts', root / 'deploy/scripts')
    (root / 'deploy/docker').mkdir()
    (root / 'deploy/docker/base-images.lock').write_text('fixture base lock')
    (root / 'VERSION').write_text('0.2.0')
    shutil.copytree(ROOT / 'examples/benchmark-plugin', root / 'examples/benchmark-plugin')
    registry = json.loads((ROOT / 'examples/benchmark-plugin/registry.json').read_text())
    shutil.copyfile(ROOT / 'examples/policy-plugin/configs/resources.yaml', root / 'configs/resources/registry.yaml')
    write_registry(root, registry)
    monkeypatch.delenv('OVLAB_PLUGIN_REGISTRY', raising=False)
    policy = {'schema_version': '0.1.0', 'kind': 'policy', 'type': 'example_policy', 'provider': 'plugin:policy:example-policy/constant', 'settings': {
        'checkpoint_id': 'example-constant', 'input': {}, 'runtime': {'quantization': 'none'},
        'action': {'action_interface': 'plugin://example-policy/interfaces/planar-v1.yaml'}, 'raw_output': {'enabled': False}}}
    path = root / 'configs/policies/example.yaml'
    path.write_text(dumps(policy))
    experiment = load(root / 'examples/benchmark-plugin/configs/evaluate.yaml')
    experiment['components']['policy'] = 'policies/example.yaml'
    experiment['components']['protocol'] = 'protocols/example.yaml'
    experiment['resources']['registry'] = 'resources/registry.yaml'
    (root / 'experiments/evaluation/example-plugin.yaml').write_text(dumps(experiment))
    protocol = load(root / 'config/protocols/smoke.yaml')
    protocol['recording']['raw_policy_output'] = False
    (root / 'configs/protocols/example.yaml').write_text(dumps(protocol))
    profile = load(root / 'config/local/example.yaml')
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
    assert prediction.actions.shape == (1, 2)
    assert (prediction.actions == 0).all()
    assert plugin.implementation.identity(caps)['method_descriptor']['id'] == 'example-constant'
    adapter.end_episode(episode)
    adapter.close()
    deployment = ModelDeployment(root, environment={})
    plan = deployment.plan('example')
    preview = deployment.run(plan, dry_run=True)
    assert preview['role'] == 'example-policy/policy-example'
    assert preview['build_command'][-1] == 'example-policy/policy-example'
    assert '/opt/ovlab/configs/plugins/registry.json' in ' '.join(preview['run_command'])
    bundle_builder = ConfigBundleBuilder(root)
    bundle_before_sync = bundle_builder.build('configs/deploy/example.yaml', None, kind='deployment')
    sync_sources(root)
    bundle = bundle_builder.build('configs/deploy/example.yaml', None, kind='deployment')
    assert bundle.sha256 == bundle_before_sync.sha256
    with bundle_builder.materialize(bundle) as config:
        document = json.loads((config / 'plugins/registry.json').read_text())
        assert document['plugins']['plugin:policy:example-policy/constant']['entrypoint'] == 'example_policy.plugin:plugin'
        assert len(document['source_identities']['example-policy/self']['content_sha256']) == 64
    assert not config.exists()
    evaluation = load(root / 'experiments/evaluation/example-plugin.yaml')
    evaluation['components']['policy'] = 'policies/example.yaml'
    evaluation['deployment'] = {'profile': 'example'}
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
    plans = build_images(root, ['policy-example'], isolated_sources=False, runner=runner)
    assert len(calls) == 1
    assert plans[0]['image'] == 'ovlab-policy-example:local'
    assert str(root / 'external/extensions/example-policy/self/Dockerfile') in calls[0]
    assert any('plugin-registry.sha256=' in value for value in calls[0])
    assert f"OVLAB_VERSION={(root / 'VERSION').read_text().strip()}" in calls[0]
    assert all(not path.exists() for path in secrets)


@pytest.mark.parametrize('kind', ['git', 'local'])
@pytest.mark.parametrize('workspace_state', ['absent', 'dirty', 'symlink'])
def test_isolated_build_combines_model_source_and_separate_adapter_without_external(
    external_plugin, tmp_path, kind, workspace_state,
):
    root = external_plugin
    model = tmp_path / 'upstream-model'
    model.mkdir()
    (model / 'model.py').write_text('VERSION = 1\n')
    git(model, 'init', '-q')
    git(model, 'config', 'user.name', 'Fixture')
    git(model, 'config', 'user.email', 'fixture@example.invalid')
    git(model, 'add', '.')
    git(model, 'commit', '-qm', 'pinned model')
    pinned = git(model, 'rev-parse', 'HEAD')
    if kind == 'git':
        (model / 'model.py').write_text('VERSION = 2\n')
        git(model, 'commit', '-qam', 'moving upstream')
    registry = registry_document(root)
    registry['sources']['model-code'] = (
        {'kind': 'git', 'url': model.as_uri(), 'commit': pinned}
        if kind == 'git' else {'kind': 'local', 'path': str(model)}
    )
    registry['plugins']['plugin:policy:example-policy/constant']['sources'].append('model-code')
    registry['builds']['example-policy/policy-example']['sources'].append('model-code')
    write_registry(root, registry)
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'OVLAB with adapter registration')
    external = root / 'external'
    if workspace_state != 'absent':
        destination = tmp_path / 'unrelated-workspace' if workspace_state == 'symlink' else external
        (destination / 'model-code').mkdir(parents=True)
        (destination / 'model-code/model.py').write_text('local work must survive')
        if workspace_state == 'symlink':
            external.symlink_to(destination, target_is_directory=True)
    contexts, secrets = [], []

    def docker(command, **kwargs):
        context = Path(command[-1])
        contexts.append(context)
        assert context != root
        assert kwargs['cwd'] == context
        assert (context / 'external/model-code/model.py').read_text() == 'VERSION = 1\n'
        assert (context / 'external/extensions/example-policy/self/src/example_policy/plugin.py').is_file()
        assert Path(command[command.index('--file') + 1]).is_relative_to(context)
        for index, value in enumerate(command):
            if value == '--secret':
                secret = Path(command[index + 1].split('src=', 1)[1])
                assert secret.is_file()
                secrets.append(secret)
        return SimpleNamespace(returncode=0)

    preview = build_images(root, ['policy-example'], dry_run=True)
    assert preview[0]['isolated_sources'] is True
    assert not contexts
    result = build_images(root, ['policy-example'], runner=docker)
    assert result[0]['isolated_sources'] is True
    # Preflight and bundling also use registered identities, independently of
    # the absent/dirty/unrelated development workspace.
    from ovlab_benchctl.plugin_identity import plugin_build_identity
    assert plugin_build_identity(root, 'policy-example') == result[0]['plugin_build_sha256']
    ConfigBundleBuilder(root).build('configs/deploy/example.yaml', None, kind='deployment')
    assert len(contexts) == 1
    assert all(not path.exists() for path in [*contexts, *secrets])
    if workspace_state == 'absent':
        assert not external.exists()
    else:
        assert (external / 'model-code/model.py').read_text() == 'local work must survive'
        assert not (external / 'example-policy').exists()
        assert external.is_symlink() == (workspace_state == 'symlink')


def test_isolated_build_cleans_context_after_docker_failure(external_plugin):
    root = external_plugin
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    contexts = []

    def fail(command, **kwargs):
        contexts.append(Path(command[-1]))
        return SimpleNamespace(returncode=1)

    with pytest.raises(RuntimeError, match='Docker build failed'):
        build_images(root, ['policy-example'], isolated_sources=True, offline=True, runner=fail)
    assert contexts and all(not path.exists() for path in contexts)
    assert not (root / 'external').exists()


def test_isolated_build_rejects_core_inputs_changed_during_preparation(external_plugin, monkeypatch):
    from ovlab_benchctl import plugin_build
    root = external_plugin
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    copy_context = plugin_build._copy_core_context
    contexts = []

    def changing_context(repository, context):
        contexts.append(context)
        copied = copy_context(repository, context)
        (root / 'deploy/docker/base-images.lock').write_text('changed after snapshot')
        return copied

    monkeypatch.setattr(plugin_build, '_copy_core_context', changing_context)
    with pytest.raises(ConfigCompatibilityError, match='input changed while preparing'):
        build_images(root, ['policy-example'], runner=lambda *args, **kwargs: pytest.fail('Docker must not run'))
    assert contexts and all(not path.exists() for path in contexts)


def test_development_external_files_do_not_change_core_build_provenance(external_plugin):
    root = external_plugin
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture')
    module = runpy.run_path(str(ROOT / 'deploy/scripts/source_manifest.py'))
    registry = registry_document(root)
    before = module['build_manifest'](root, registry=registry, source_rows=[])
    (root / 'external/unrelated').mkdir(parents=True)
    (root / 'external/unrelated/local.py').write_text('unrelated development work')
    after = module['build_manifest'](root, registry=registry, source_rows=[])
    assert after['source_content_sha256'] == before['source_content_sha256']
    assert after['dirty'] == before['dirty']


def test_isolated_offline_build_reads_pinned_git_objects_without_dirty_checkout_files(external_plugin, tmp_path):
    root = external_plugin
    upstream = tmp_path / 'remote'
    upstream.mkdir()
    (upstream / 'model.py').write_text('PINNED = True\n')
    git(upstream, 'init', '-q')
    git(upstream, 'config', 'user.name', 'Fixture')
    git(upstream, 'config', 'user.email', 'fixture@example.invalid')
    git(upstream, 'add', '.')
    git(upstream, 'commit', '-qm', 'model')
    pinned = git(upstream, 'rev-parse', 'HEAD')
    registry = registry_document(root)
    registry['sources']['model-code'] = {'kind': 'git', 'url': upstream.as_uri(), 'commit': pinned}
    registry['builds']['example-policy/policy-example']['sources'].append('model-code')
    write_registry(root, registry)
    sync_sources(root, ['model-code'])
    cached = root / 'external/model-code'
    (cached / 'model.py').write_text('dirty development changes')
    (cached / 'untracked.py').write_text('must not enter image')
    shutil.rmtree(upstream)
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', 'VERSION')
    git(root, 'commit', '-qm', 'fixture')
    status = git(cached, 'status', '--porcelain')

    def docker(command, **kwargs):
        source = Path(command[-1]) / 'external/model-code'
        assert (source / 'model.py').read_text() == 'PINNED = True\n'
        assert not (source / 'untracked.py').exists()
        assert git(source, 'rev-parse', 'HEAD') == pinned
        return SimpleNamespace(returncode=0)

    result = build_images(root, ['policy-example'], offline=True, runner=docker)
    assert result[0]['isolated_sources'] is True
    assert git(cached, 'status', '--porcelain') == status
    assert (cached / 'model.py').read_text() == 'dirty development changes'
    assert not (root / 'external/extensions/example-policy/self').exists()


def test_bundled_experiment_resolves_and_serves_without_git_or_source_inspection(external_plugin, monkeypatch):
    from ovlab_benchctl.application import OvlabApplication
    from ovlab_benchctl import plugin_identity, plugin_sources
    import ovlab_remote_policy.service

    root = external_plugin
    registry = registry_document(root)
    remote = {'kind': 'git', 'url': 'https://example.invalid/model.git', 'commit': 'a' * 40}
    registry['sources']['remote-model'] = remote
    registry['plugins']['plugin:policy:example-policy/constant']['sources'].append('remote-model')
    registry['builds']['example-policy/policy-example']['sources'].append('remote-model')
    write_registry(root, registry)
    experiment = 'experiments/evaluation/example-plugin.yaml'
    resolver = ConfigResolver(root / 'configs', repository_root=root)
    host = resolver.resolve(experiment, local_profile='configs/local/example.yaml', environment={})
    builder = ConfigBundleBuilder(root)
    bundle = builder.build(experiment, None)
    # Image source trees have no checkout metadata. A local project can also
    # differ from the frozen bundle after it was built on the host.
    copied = root / 'external/remote-model'
    copied.mkdir(parents=True)
    (copied / 'model.py').write_text('# copied into the image without .git\n')
    (root / 'examples/policy-plugin/README.md').write_text('changed after bundling')

    def no_inspection(*args, **kwargs):
        pytest.fail('a bundled worker must not inspect source trees or invoke Git')

    monkeypatch.setattr(plugin_sources, '_git', no_inspection)
    monkeypatch.setattr(plugin_identity, 'inspect_sources', no_inspection)
    captured = {}

    class Service:
        def __init__(self, socket, adapter, *, identity_provider, startup_context, **kwargs):
            self.adapter, self.identity_provider, self.startup_context = adapter, identity_provider, startup_context

        def serve(self):
            capabilities = self.adapter.initialize(self.startup_context)
            captured.update(self.identity_provider(capabilities))
            self.adapter.close()

    monkeypatch.setattr(ovlab_remote_policy.service, 'PolicyService', Service)
    with builder.materialize(bundle) as config:
        worker_root = root / 'isolated-worker'
        shutil.copytree(config, worker_root / 'configs')
        config = worker_root / 'configs'
        environment = {
            'OVLAB_PLUGIN_REGISTRY': str(config / 'plugins/registry.json'),
            'OVLAB_LOCAL_PROFILE': str(config / 'plugins/runtime-profile.yaml'),
        }
        worker = ConfigResolver(config, repository_root=worker_root, environment=environment).resolve(
            config / experiment, local_profile=environment['OVLAB_LOCAL_PROFILE'],
        )
        assert worker.scientific_config == host.scientific_config
        assert worker.scientific_config_hash == host.scientific_config_hash
        assert worker.scientific_config['policy_plugin']['sources']['remote-model'] == remote
        # Exercise the evaluation service path too; standalone deploy uses a
        # different resolver and did not expose this regression.
        app = OvlabApplication(worker_root, environment=environment)
        result = app.serve(config / experiment, host='127.0.0.1', port=8123)
        assert result['closed'] is True
        assert captured['method_descriptor']['id'] == 'example-constant'


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
    assert captured['method_descriptor']['id'] == 'example-constant'


def test_listing_and_policy_deploy_planning_need_only_standard_library(external_plugin):
    env = dict(os.environ, OVLAB_ROOT=str(external_plugin))
    # Explicitly remove site-packages and test helper paths.
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in (ROOT / 'code').glob('**/src'))
    completed = subprocess.run([sys.executable, '-S', '-m', 'ovlab_benchctl.cli', 'deploy', 'run', 'example', '--dry-run', '--json'], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)['result']['role'] == 'example-policy/policy-example'


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
            IMAGE_CONTRACT_LABEL: 'policy-rpc-v2-config-bundle-v3',
            'io.github.ovlab.service.transport': 'tcp-persistent-v1',
            'io.github.ovlab.role': 'example-policy/policy-example', PLUGIN_BUILD_LABEL: '0'*64}}}]), stderr='')

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
    plugin = registry['plugins']['plugin:policy:example-policy/constant']
    if mutation == 'omitted-source': registry['builds']['example-policy/policy-example']['sources'] = []
    if mutation == 'different-image': plugin['deployment']['image'] = 'different:local'
    if mutation == 'reserved-service': plugin['deployment']['policy_service'] = 'reporting'
    if mutation == 'recipe-escape': registry['builds']['example-policy/policy-example']['dockerfile'] = '../outside/Dockerfile'
    if mutation == 'duplicate-document':
        plugin['document_kinds'] = ['custom_descriptor']
        registry['plugins']['second'] = deepcopy(plugin)
        registry['plugins']['second'].update(id='second', config_type='second', provider_ref='plugin:policy:second/default')
    write_registry(root, registry)
    with pytest.raises(ConfigSchemaError, match=message):
        registry_document(root)
    assert not (root/'external').exists()


def test_explicit_python_environment_selects_registry_without_process_mutation(external_plugin, monkeypatch):
    from ovlab_benchctl.application import OvlabApplication
    root = external_plugin
    registry = registry_document(root)
    registry['builds']['example-policy/policy-example']['image'] = 'example-override:local'
    registry['plugins']['plugin:policy:example-policy/constant']['deployment']['image'] = 'example-override:local'
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
        (root/'external/extensions/example-policy/self/src/example_policy/runtime.py').write_text('# changed during build')
        return SimpleNamespace(returncode=0)

    with pytest.raises(ConfigCompatibilityError, match='changed during build'):
        build_images(root, ['policy-example'], isolated_sources=False, runner=docker)
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
    write_registry(root, json.loads((ROOT/'examples/policy-plugin/registry.json').read_text()))
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
    baseline = tree_identity(snapshot)
    (snapshot/'toy.egg-info').mkdir()
    (snapshot/'toy.egg-info/PKG-INFO').write_text('generated')
    (snapshot/'toy.dist-info').mkdir()
    (snapshot/'toy.dist-info/METADATA').write_text('generated')
    (snapshot/'experiments/rollouts/one').mkdir(parents=True)
    (snapshot/'experiments/rollouts/one/episode.mp4').write_bytes(b'video')
    assert sync_sources(root, offline=True)[0]['dirty'] is False
    assert tree_identity(snapshot) == baseline
    (snapshot/'experiments/rollouts/one/analysis.py').write_text('source code')
    with pytest.raises(ConfigCompatibilityError, match='analysis.py'):
        sync_sources(root, offline=True)
    (snapshot/'experiments/rollouts/one/analysis.py').unlink()
    (snapshot/'ignored.py').write_text('UNPINNED = True')
    with pytest.raises(ConfigCompatibilityError, match='local changes'):
        sync_sources(root, offline=True)
    assert inspect_sources(root)[0]['dirty'] is True


def test_local_snapshot_excludes_generated_metadata_and_rollout_video(tmp_path):
    root, original = tmp_path/'ovlab', tmp_path/'project'
    original.mkdir()
    (original/'runtime.py').write_text('source')
    write_registry(root, {'schema_version': 'ovlab.plugins/v1',
                          'sources': {'toy': {'kind': 'local', 'path': str(original)}},
                          'plugins': {}, 'builds': {}})
    before = tree_identity(original)
    (original/'toy.egg-info').mkdir()
    (original/'toy.egg-info/PKG-INFO').write_text('generated')
    (original/'rollouts/day').mkdir(parents=True)
    (original/'rollouts/day/result.mp4').write_bytes(b'video')
    (original/'rollouts/day/analysis.py').write_text('important source')
    assert tree_identity(original) != before
    sync_sources(root)
    snapshot = root/'external/toy'
    assert not (snapshot/'toy.egg-info').exists()
    assert not (snapshot/'rollouts/day/result.mp4').exists()
    assert (snapshot/'rollouts/day/analysis.py').read_text() == 'important source'


def test_tracked_generated_files_are_never_overwritten(tmp_path):
    project = tmp_path/'upstream'
    project.mkdir()
    git(project, 'init', '-q')
    git(project, 'config', 'user.name', 'Fixture')
    git(project, 'config', 'user.email', 'fixture@example.invalid')
    (project/'rollouts').mkdir()
    (project/'rollouts/tracked.mp4').write_bytes(b'original')
    git(project, 'add', '.')
    git(project, 'commit', '-qm', 'fixture')
    root = tmp_path/'ovlab'
    write_registry(root, {'schema_version': 'ovlab.plugins/v1',
                          'sources': {'toy': {'kind': 'git', 'url': str(project),
                                              'commit': git(project, 'rev-parse', 'HEAD')}},
                          'plugins': {}, 'builds': {}})
    sync_sources(root)
    video = root/'external/toy/rollouts/tracked.mp4'
    video.write_bytes(b'local modification')
    with pytest.raises(ConfigCompatibilityError, match='tracked.mp4'):
        sync_sources(root, offline=True)
    assert video.read_bytes() == b'local modification'


def test_import_cache_cannot_silently_reuse_a_different_local_checkout(external_plugin):
    root = external_plugin
    original = plugin_for_type('example_policy', root)
    assert original.implementation is original.implementation
    second = root/'other-project'
    shutil.copytree(root/'examples/policy-plugin', second)
    (second/'src/example_policy/plugin.py').write_text((second/'src/example_policy/plugin.py').read_text()+'\n# distinct implementation\n')
    registry = registry_document(root)
    registry['sources']['example-policy/self']['path'] = str(second)
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
