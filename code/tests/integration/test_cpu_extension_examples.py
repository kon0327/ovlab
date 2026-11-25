"""Independent example projects exercise SDKs, public planning and canonical output."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def example_root(tmp_path):
    root = tmp_path / 'checkout'
    for relative in ('config', 'examples/policy-plugin', 'examples/benchmark-plugin'):
        shutil.copytree(ROOT / relative, root / relative, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for relative in ('deploy/compose/compose.yaml', 'deploy/docker/Dockerfile.reporting',
                     'deploy/docker/install-ovlab-packages.sh', 'deploy/locks/reporting.pylock.toml',
                     'deploy/locks/reporting.requirements.txt', 'VERSION'):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    configure = runpy.run_path(str(root / 'examples/benchmark-plugin/configure.py'))['configure']
    configure(root, data_root=tmp_path / 'data')
    return root


def environment(root):
    values = dict(os.environ)
    values.update(
        OVLAB_ROOT=str(root),
        OVLAB_PYTHON=sys.executable,
        OVLAB_EXTENSION_REGISTRY=str(root / 'configs/extensions/cpu-example.json'),
        OVLAB_LOCAL_PROFILE=str(root / 'configs/local/cpu-plugin-example.yaml'),
        OVLAB_COMPOSE_ENV_FILE=str(root / 'configs/local/cpu-plugin-example.env'),
        OVLAB_PROGRESS='off',
    )
    values.pop('OVLAB_PLUGIN_REGISTRY', None)
    return values


def test_configure_is_idempotent_and_preserves_registration_and_edits(example_root):
    root = example_root
    configure = runpy.run_path(str(root / 'examples/benchmark-plugin/configure.py'))['configure']
    data = root.parent / 'data'
    sentinel = root / 'plugins/registry.json'
    sentinel.parent.mkdir()
    sentinel.write_text('{"preserve": true}\n')
    before = sentinel.read_bytes()
    expected = configure(root, data_root=data, dry_run=True)
    assert len(expected) == 4 and configure(root, data_root=data) == expected
    assert sentinel.read_bytes() == before
    experiment = root / 'configs/evaluation/cpu-plugin-example.yaml'
    experiment.write_text(experiment.read_text() + '# user edit\n')
    with pytest.raises(ValueError, match='preserving modified'):
        configure(root, data_root=data)


def test_example_control_entrypoints_and_deployment_hook_are_stdlib_only():
    script = '''
import importlib.abc, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("policy-plugin", "benchmark-plugin"):
    sys.path.insert(0, str(root / "examples" / name / "src"))
blocked = {"numpy", "torch", "libero", "mujoco", "robosuite", "ovlab_core", "ovlab_runner", "ovlab_benchctl"}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked or fullname.endswith(".runtime"):
            raise AssertionError("runtime import: " + fullname)
sys.meta_path.insert(0, Reject())
from example_policy.plugin import plugin as policy
from example_benchmark.plugin import plugin as benchmark
document = {"type": "example_planar", "settings": {"task_ids": "all", "action": {"interface_ref": "fixture"}}}
spec = benchmark.resolve_deployment(document, execution_profile={}, local_execution={}, devices={}, environment={}, deployment={})
assert not spec.requires_gpu and not spec.environment and not spec.mounts
assert policy.api_version == "2"
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpu_examples_resolve_freeze_run_metrics_and_current_report(example_root):
    script = '''
import os, sys
from pathlib import Path
from ovlab_benchctl import ConfigResolver
from ovlab_benchctl.benchmark_plugins import benchmark_for_document, create_adapter
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.policy_plugins import registry_scope, plugin_for_type
from ovlab_core.contracts import RunContext, RunId
from ovlab_runner import ExperimentRunner, FilesystemRunArtifactStore, DeterministicClock, verify_run, recompute_run_metrics
from ovlab_runner.reporting import build_run_report
root = Path(sys.argv[1])
with registry_scope(root, dict(os.environ)):
    builder = ConfigBundleBuilder(root)
    document = builder.load_component("evaluation/cpu-plugin-example.yaml", "experiment")
    benchmark_doc = builder.load_component(document["components"]["benchmark"], "benchmark")
    policy_doc = builder.load_component(document["components"]["policy"], "policy")
    resolved = ConfigResolver(root / "configs", repository_root=root).resolve(
        root / "configs/evaluation/cpu-plugin-example.yaml", local_profile=os.environ["OVLAB_LOCAL_PROFILE"], environment=dict(os.environ))
    assert resolved.action_spec.dimension == 2
    ids = resolved.benchmark_context.task_set.task_ids
    assert tuple(map(str, ids)) == ("calibration:alpha", "reach:omega", "stability@hold")
    bundle = builder.build("evaluation/cpu-plugin-example.yaml", None)
    assert any(item["path"].endswith("task_catalog.json") for item in bundle.files)
    with builder.materialize(bundle) as frozen:
        assert (frozen / ".ovlab-bundle.json").is_file()
    owner = benchmark_for_document(benchmark_doc, root)
    benchmark = create_adapter(owner, resolved.benchmark_settings)
    policy_owner = plugin_for_type(policy_doc["type"], root, provider=policy_doc["provider"])
    policy = policy_owner.implementation.create_adapter(resolved.policy_settings)
    context = RunContext(RunId("cpu-example-integration"), 1, "CPU fixture", 7)
    plan = resolved.create_plan(context, ids)
    store = FilesystemRunArtifactStore(root.parent / "canonical-runs")
    runner = ExperimentRunner(plan, benchmark, policy, store, clock=DeterministicClock(),
                              configuration_snapshot=resolved.configuration_snapshot())
    try:
        runner.connect()
        runner.run()
    finally:
        runner.close()
    path = store._run_path(context.run_id)
    assert verify_run(path)["verified_episode_count"] == 3
    assert recompute_run_metrics(path)["all_results_agree"]
    from ovlab_runner import build_report_model
    derived = build_report_model(path)
    statistics = derived['tasks'][0]['episodes'][0]['trace_view']['action_component_statistics']
    assert set(statistics) == {'joint_0', 'joint_1'}
    report = build_run_report(path)
    assert report["status"] == "completed" and len(report["episodes"]) == 3
    assert {item["terminal_status"] for item in report["episodes"]} == {"success", "time_limit"}
'''
    result = subprocess.run([sys.executable, '-c', script, str(example_root)],
                            env=environment(example_root), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_cli_cpu_compose_plan_has_no_native_runtime_requirements(example_root):
    result = subprocess.run([str(ROOT / 'ovlab'), 'eval', 'run',
        'configs/evaluation/cpu-plugin-example.yaml', '--offline', '--dry-run', '--json'],
        env=environment(example_root), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)['result']
    assert document['side_effects_performed'] is False
    assert document['benchmark_provider'] == 'plugin:benchmark:example-benchmark/planar'
    assert document['renderer'] == ''


def test_frozen_benchmark_control_dependencies_and_selection_are_closed(example_root, tmp_path, monkeypatch):
    from ovlab_benchctl.config_bundle import ConfigBundleBuilder
    from ovlab_benchctl.policy_plugins import registry_scope
    root = example_root
    support = root / 'separate-control-project'
    support.mkdir()
    (support / 'fixture_support.py').write_text('VALUE = "only selected code"\n')
    project = root / 'examples/benchmark-plugin'
    manifest_path = project / 'extension.json'
    manifest = json.loads(manifest_path.read_text())
    provider = manifest['providers']['planar']
    manifest['sources']['control-support'] = {'kind':'local', 'path':str(support)}
    provider['sources'] = ['control-support']
    provider['control_sources'] = ['control-support']
    manifest['builds']['benchmark-example']['sources'] = ['control-support']
    manifest['providers']['unselected'] = {**provider, 'id':'unselected'}
    manifest_path.write_text(json.dumps(manifest))
    implementation = project / 'src/example_benchmark/plugin.py'
    implementation.write_text('from fixture_support import VALUE\n' + implementation.read_text())
    for name in list(sys.modules):
        if name == 'fixture_support' or name.startswith(('example_benchmark', 'example_policy')):
            monkeypatch.delitem(sys.modules, name)
    values = environment(root)
    with registry_scope(root, values):
        builder = ConfigBundleBuilder(root)
        bundle = builder.build('evaluation/cpu-plugin-example.yaml', None)
        with builder.materialize(bundle) as frozen:
            worker = tmp_path / 'worker'
            shutil.copytree(frozen, worker / 'configs')
    support.rename(root / 'unavailable-control-project')
    (root / 'examples').rename(root / 'unavailable-examples')
    values.update(OVLAB_ROOT=str(worker),
                  OVLAB_EXTENSION_REGISTRY=str(worker / 'configs/extensions/registry.json'),
                  OVLAB_PLUGIN_REGISTRY=str(worker / 'configs/plugins/registry.json'),
                  PYTHONPATH=os.pathsep.join(str(path) for path in ROOT.glob('code/**/src')))
    script = '''
from pathlib import Path
import sys
from ovlab_benchctl.benchmark_plugins import extension_catalog, benchmark_for_document
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.errors import ConfigReferenceError
root = Path(sys.argv[1])
builder = ConfigBundleBuilder(root)
document = builder.load_component('plugin://example-benchmark/benchmark.yaml', 'benchmark')
owner = benchmark_for_document(document, root)
assert owner.implementation.api_version == '1'
import fixture_support
assert Path(fixture_support.__file__).is_relative_to(root)
catalog = extension_catalog(root)
try: catalog.resolve_provider('plugin:benchmark:example-benchmark/unselected')
except ConfigReferenceError: pass
else: raise AssertionError('unselected provider escaped frozen selection')
'''
    completed = subprocess.run([sys.executable, '-c', script, str(worker)], env=values, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    missing = worker / 'configs/plugins/projects/example-benchmark/src/example_benchmark/plugin.py'
    missing.unlink()
    # Ambient host code must not rescue an incomplete frozen entrypoint.
    values['PYTHONPATH'] += os.pathsep + str(root / 'unavailable-examples/benchmark-plugin/src')
    completed = subprocess.run([sys.executable, '-c', script, str(worker)], env=values, text=True, capture_output=True)
    assert completed.returncode != 0 and 'entrypoint is missing' in completed.stderr
