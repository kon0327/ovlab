"""Each model is an independently packaged, registered integration."""
from email.parser import Parser
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROJECTS = {
    'openvla': ('vanilla', 'lora_merged'),
    'openvla-oft': ('oft',),
    'openvla-quic': ('quic',),
}
SUPPORT = ROOT / 'plugins/_shared/openvla-support'


@pytest.mark.parametrize('name,namespaces', PROJECTS.items())
def test_model_wheel_owns_only_its_model_and_complete_project_assets(tmp_path, name, namespaces):
    original = ROOT / 'plugins' / name
    project = tmp_path / 'project'
    shutil.copytree(original, project, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build', 'dist'))
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                             '--disable-pip-version-check', '--wheel-dir', str(tmp_path), str(project)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = tmp_path.glob(f"ovlab_{name.replace('-', '_')}_plugin-*.whl")
    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
        for namespace in namespaces:
            assert f'ovlab_openvla_{namespace}/__init__.py' in members
        for other in {'vanilla', 'lora_merged', 'oft', 'quic', 'common', 'training'} - set(namespaces):
            assert f'ovlab_openvla_{other}/__init__.py' not in members
        assert not any(path.startswith(('ovlab_core/', 'ovlab_benchctl/')) for path in members)
        prefix = f'.data/data/share/ovlab/plugins/{name}/'
        for asset in ('plugin.json', 'pyproject.toml', 'setup.py', 'configs/registry/default.yaml',
                      'examples/policy.yaml', 'scripts/install-plugin.sh'):
            assert any(prefix + asset in path for path in members), asset
        metadata_name, = (path for path in members if path.endswith('.dist-info/METADATA'))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    assert metadata['Name'] == f'ovlab-{name}-plugin'
    assert metadata['Version'] == json.loads((original / 'plugin.json').read_text())['version']
    assert metadata.get_all('Requires-Dist') == ['ovlab-openvla-support==0.2.0']

    installed = tmp_path / 'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
                    '--disable-pip-version-check', '--target', str(installed), str(wheel)],
                   check=True, text=True, capture_output=True)
    script = ("from importlib.metadata import entry_points; "
              f"entry = next(e for e in entry_points(group='ovlab.plugin_projects') if e.name == {name!r}); "
              "print(entry.load()())")
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(installed)},
                            check=True, text=True, capture_output=True)
    assert (Path(result.stdout.strip()) / 'configs/registry/default.yaml').is_file()


def test_shared_support_is_a_library_and_builds_declare_it_explicitly():
    assert not (SUPPORT / 'plugin.json').exists()
    assert {path.name for path in (SUPPORT / 'src').iterdir() if path.is_dir() and not path.name.endswith('.egg-info')} == {
        'ovlab_openvla_common', 'ovlab_openvla_training'}
    for name in PROJECTS:
        project = ROOT / 'plugins' / name
        manifest = json.loads((project / 'plugin.json').read_text())
        assert (project / manifest['sources']['openvla-support']['path']).resolve() == SUPPORT.resolve()
        for descriptor in manifest['providers'].values():
            assert 'openvla-support' in descriptor['sources']
            assert 'openvla-support' in descriptor['control_sources']
        for recipe in manifest['builds'].values():
            assert 'openvla-support' in recipe['sources']
            dockerfile = (project / recipe['dockerfile']).read_text()
            assert f'COPY external/extensions/{name}/self /opt/ovlab/external/extensions/{name}/self' in dockerfile
            assert f'COPY external/extensions/{name}/openvla-support /opt/ovlab/external/extensions/{name}/openvla-support' in dockerfile
            assert f'/opt/ovlab/external/extensions/{name}/openvla-support' in dockerfile
            assert f'bash /opt/ovlab/external/extensions/{name}/self/scripts/install-plugin.sh' in dockerfile
            for lock in recipe['locks']:
                assert (project / lock).is_file(), lock
    installer = (ROOT / 'deploy/docker/install-ovlab-packages.sh').read_text()
    assert 'openvla' not in installer


@pytest.mark.parametrize('name,policy_type', [('openvla','openvla_vanilla'),
                                             ('openvla-oft','openvla_oft'),
                                             ('openvla-quic','openvla_quic')])
def test_each_model_loads_with_only_its_own_registration(tmp_path, name, policy_type):
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema_version':'ovlab.plugins/v2', 'plugins':{
        name:{'source':{'kind':'local','path':str(ROOT / 'plugins' / name)}}}}))
    paths = [str(path) for path in (ROOT / 'code').glob('**/src')]
    script = f'''import sys
from ovlab_benchctl.policy_plugins import registry_document, plugin_for_type
from ovlab_benchctl.training_plugins import training_registration
root={str(ROOT)!r}
registry=registry_document(root)
assert set(registry['projects']) == {{{name!r}}}
plugin=plugin_for_type({policy_type!r}, root)
assert plugin.implementation.config_type == {policy_type!r}
assert 'torch' not in sys.modules
if {name!r} != 'openvla-oft':
    implementation='openvla-reference' if {name!r} == 'openvla' else 'openvla-quic'
    owner, declaration=training_registration(implementation, root)
    assert owner.descriptor['project'] == {name!r}
'''
    subprocess.run([sys.executable,'-c',script], cwd=tmp_path, check=True, text=True, capture_output=True,
                   env={**os.environ,'PYTHONPATH':os.pathsep.join(paths), 'OVLAB_PLUGIN_REGISTRY':str(registry)})


def test_plugin_sources_use_public_contracts_without_importing_cli_internals():
    for project in [SUPPORT, *(ROOT / 'plugins' / name for name in PROJECTS)]:
        for path in (project / 'src').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [
                    alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                assert not any(name == 'ovlab_benchctl' or name.startswith('ovlab_benchctl.') for name in names), path


def test_pinned_control_dependency_uses_isolated_cache_and_frozen_workers_need_no_git(tmp_path):
    source = tmp_path / 'support-git'
    source.mkdir()
    (source / 'library.py').write_text('VERSION = 1\n')
    for arguments in [('init','-q'), ('config','user.name','Fixture'),
                      ('config','user.email','fixture@example.invalid'), ('add','.'), ('commit','-qm','initial')]:
        subprocess.run(['git','-C',str(source),*arguments], check=True, capture_output=True)
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    paths = [str(path) for path in (ROOT / 'code').glob('**/src')]
    script = f'''from pathlib import Path
import os
from ovlab_benchctl.plugin_projects import control_source_root
from ovlab_benchctl.errors import ConfigCompatibilityError
root=Path({str(tmp_path / 'workspace')!r})
external=root/'external/support'
external.mkdir(parents=True)
(external/'local-edit.py').write_text('preserve me')
registry={{'sources':{{'support':{{'kind':'git','url':{str(source)!r},'commit':{revision!r}}}}}}}
resolved=control_source_root('support',root,registry)
assert resolved.is_relative_to(root/'plugins/.cache/control')
assert (resolved/'library.py').read_text()=='VERSION = 1\\n'
assert (external/'local-edit.py').read_text()=='preserve me'
assert control_source_root('support',root,registry,acquire=False)==resolved
(resolved/'library.py').write_text('edited')
try:
    control_source_root('support',root,registry)
except ConfigCompatibilityError:
    pass
else:
    raise AssertionError('modified dependency cache was accepted')
frozen=root/'configs/plugins/sources/support'
frozen.mkdir(parents=True)
(frozen/'library.py').write_text('frozen')
(root/'configs/.ovlab-bundle.json').write_text('{{}}')
os.environ['PATH']=''
assert control_source_root('support',root,registry)==frozen
'''
    subprocess.run([sys.executable,'-c',script], cwd=tmp_path, check=True, text=True, capture_output=True,
                   env={**os.environ,'PYTHONPATH':os.pathsep.join(paths)})


def test_quic_training_template_never_supplies_fake_adapter_identity():
    from copy import deepcopy
    from ovlab_benchctl.config_bundle import ConfigBundleBuilder
    from ovlab_benchctl.policy_plugins import plugin_for_type
    from ovlab_policy_sdk.plugin import PolicyPluginValidationError

    document = ConfigBundleBuilder(ROOT).load_component('plugin://openvla-quic/policy/default.yaml',
                                                        'policy', validate_document=False)
    implementation = plugin_for_type('openvla_quic', ROOT).implementation
    assert not {'adapter_path','adapter_manifest_sha256','adapter_tensor_sha256'} & document['settings']['quic'].keys()
    implementation.validate_checkpoint_template(document, 'template')
    with pytest.raises(PolicyPluginValidationError, match='adapter'):
        implementation.validate_policy(document, 'runtime')
    partial = deepcopy(document)
    partial['settings']['quic']['adapter_path'] = '/explicit/adapter'
    with pytest.raises(PolicyPluginValidationError, match='identity'):
        implementation.validate_checkpoint_template(partial, 'partial')
    invalid = deepcopy(document)
    invalid['settings']['runtime']['quantization'] = '4bit'
    with pytest.raises(PolicyPluginValidationError, match='quantization'):
        implementation.validate_checkpoint_template(invalid, 'quantized')


def test_remote_model_project_acquires_its_declared_shared_library_without_development_external(tmp_path):
    def commit(path):
        for arguments in [('init','-q'),('config','user.name','Fixture'),
                          ('config','user.email','fixture@example.invalid'),('add','.'),('commit','-qm','fixture')]:
            subprocess.run(['git','-C',str(path),*arguments],check=True,capture_output=True)
        return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()

    support = tmp_path / 'support-git'
    ignore = shutil.ignore_patterns('__pycache__','*.egg-info','build','dist')
    shutil.copytree(SUPPORT, support, ignore=ignore)
    support_revision = commit(support)
    model = tmp_path / 'model-git'
    shutil.copytree(ROOT / 'plugins/openvla-quic',model,ignore=ignore)
    manifest = json.loads((model/'plugin.json').read_text())
    manifest['sources']['openvla-support']={'kind':'git','url':support.as_uri(),'commit':support_revision}
    (model/'plugin.json').write_text(json.dumps(manifest))
    model_revision = commit(model)
    workspace = tmp_path/'workspace'
    (workspace/'plugins').mkdir(parents=True)
    (workspace/'plugins/registry.json').write_text(json.dumps({'schema_version':'ovlab.plugins/v2', 'plugins':{
        'openvla-quic':{'source':{'kind':'git','url':model.as_uri(),'commit':model_revision}}}}))
    script = f'''from pathlib import Path
from ovlab_benchctl.policy_plugins import plugin_for_type
root=Path({str(workspace)!r})
implementation=plugin_for_type('openvla_quic',root).implementation
assert implementation.config_type=='openvla_quic'
assert not (root/'external').exists()
assert (root/'plugins/.cache/control/openvla-support'/{support_revision!r}/'external/openvla-support/src/ovlab_openvla_common').is_dir()
'''
    paths=[str(path) for path in (ROOT/'code').glob('**/src')]
    subprocess.run([sys.executable,'-c',script],cwd=tmp_path,check=True,text=True,capture_output=True,
                   env={**os.environ,'PYTHONPATH':os.pathsep.join(paths),'OVLAB_ROOT':str(workspace),
                        'OVLAB_PLUGIN_REGISTRY':str(workspace/'plugins/registry.json')})
