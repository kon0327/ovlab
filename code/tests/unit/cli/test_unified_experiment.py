"""One experiment owner for source configurations and persisted run trees."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.experiment_references import find_experiment
from ovlab_benchctl.experiment_workflow import (
    ExperimentIntegrityError, ExperimentSpecError, ExperimentWorkflow, _spec_hash,
)
from ovlab_benchctl.report_auto import generate_report
from ovlab_benchctl.run_references import RunReferenceAmbiguousError, RunReferenceError
from ovlab_benchctl.strict_yaml import dumps
from ovlab_benchctl.training_composition import load_training_profile
from ovlab_runner import (
    DeterministicClock, ExperimentRunner, FilesystemRunArtifactStore,
    RunConfigurationSnapshot, RunIntegrityError,
)

ROOT = Path(__file__).resolve().parents[4]


def _hash_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('source,phase', [
    ('experiments/training/openvla-libero10-quic-smoke.yaml', 'train'),
    ('experiments/evaluation/libero10-lora-4bit-three-per-task.yaml', 'evaluate'),
])
def test_standalone_sources_share_the_tree_plan_and_remain_unchanged(tmp_path, source, phase):
    workflow = ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(tmp_path / 'data')})
    source_path = ROOT / source
    before = source_path.read_bytes()
    first = workflow.plan(source)
    assert first == workflow.plan(source)
    assert first['steps'] == [{'node': '', 'phase': phase}]
    assert first['nodes'][0]['execute'] == [phase]
    assert first['nodes'][0]['configuration'][phase]['document']['kind'] in {'training', 'experiment'}
    assert workflow.verify(source)['source_type'] == 'configuration'
    assert source_path.read_bytes() == before
    assert not (tmp_path / 'data').exists()


@pytest.mark.parametrize('phase', ['train', 'evaluate'])
def test_external_standalone_yaml_resolves_extends_before_normalizing(tmp_path, phase):
    if phase == 'train':
        document = dict(load_training_profile(ROOT / 'experiments/training/openvla-libero10-quic-smoke.yaml').document)
        child = {'kind': 'training', 'extends': 'base.yaml', 'training': {'max_steps': 2}}
    else:
        document = ConfigBundleBuilder(ROOT).load_component('experiments/evaluation/libero10-lora-4bit-three-per-task.yaml', 'experiment')
        child = {'kind': 'experiment', 'extends': 'base.yaml', 'experiment': {'name': 'Inherited external evaluation'}}
    (tmp_path / 'base.yaml').write_text(dumps(document))
    source = tmp_path / 'child.yaml'
    source.write_text(dumps(child))
    plan = ExperimentWorkflow(ROOT).plan(source)
    resolved = plan['nodes'][0]['configuration'][phase]['document']
    assert 'extends' not in resolved
    if phase == 'train':
        assert resolved['training']['max_steps'] == 2
    else:
        assert resolved['experiment']['name'] == 'Inherited external evaluation'


def test_bare_experiment_names_are_unique_across_all_source_types():
    workflow = ExperimentWorkflow(ROOT)
    assert workflow.plan('openvla-libero10-quic-smoke')['steps'] == [{'node': '', 'phase': 'train'}]
    with pytest.raises(ExperimentSpecError, match='ambiguous'):
        workflow.plan('example')
    with pytest.raises(ExperimentSpecError, match='kind'):
        workflow.plan('configs/policies/openvla-quic/quic-peft-bones.yaml')


def _tree():
    return {'schema_version': 'ovlab.experiment/v1', 'kind': 'workflow', 'id': 'study', 'name': 'Study',
            'execution': {'max_workers': 1},
            'config': {'train': {'profile': 'openvla-libero10-quic-smoke.yaml', 'patch': {'training': {'max_steps': 2}}}},
            'children': [{'id': 'a', 'execute': ['train']},
                         {'id': 'b', 'execute': ['train'], 'config': {'train': {'patch': {'training': {'seed': 99}}}}}]}


def test_execution_settings_and_configuration_inherit_from_yaml(tmp_path):
    source = tmp_path / 'tree.yaml'
    source.write_text(dumps(_tree()))
    plan = ExperimentWorkflow(ROOT).plan(source)
    assert plan['execution'] == {'mode': 'sequential', 'max_workers': 1}
    for node in plan['nodes'][1:]:
        assert node['configuration']['train']['document']['training']['max_steps'] == 2
    assert plan['nodes'][2]['configuration']['train']['document']['training']['seed'] == 99
    assert plan['nodes'][1]['configuration']['train']['document']['training']['seed'] != 99


def test_two_task_training_smoke_binds_training_outputs_and_keeps_oft_reference():
    workflow = ExperimentWorkflow(ROOT)
    plan = workflow.plan('experiments/workflows/libero10-two-task-training-smoke.yaml')
    assert plan['execution']['max_workers'] == 1
    nodes = {node['path']: node for node in plan['nodes']}
    expected = {
        'lora/full': ('openvla_lora_merged', 'none'),
        'lora/eight-bit': ('openvla_lora_merged', '8bit'),
        'lora/four-bit': ('openvla_lora_merged', '4bit'),
        'quic/full': ('openvla_quic', 'none'),
        'oft/full': ('openvla_oft', 'none'),
        'oft/eight-bit': ('openvla_oft', '8bit'),
        'oft/four-bit': ('openvla_oft', '4bit'),
    }
    assert len(plan['steps']) == len(expected) + 2
    for node_path in ('lora', 'quic'):
        profile = nodes[node_path]['configuration']['train']['document']
        assert nodes[node_path]['execute'] == ['train']
        assert profile['training']['max_steps'] == 3
        assert len(profile['dataset']['partition']['groups']) == 2
    for node_path, (policy_type, quantization) in expected.items():
        node = nodes[node_path]
        configuration = node['configuration']['evaluate']
        assert node['execute'] == ['evaluate']
        benchmark = configuration['component_documents']['benchmark']['document']
        assert benchmark['settings']['task_indices'] == [0, 1]
        protocol = workflow.bundle.load_component(configuration['document']['components']['protocol'], 'protocol')
        assert protocol['execution']['rollouts_per_task'] == 1
        assert protocol['execution']['maximum_episode_steps'] == 520
        policy = configuration['component_documents'].get('policy', {}).get('document')
        if policy is None:
            policy = workflow.bundle.load_component(configuration['document']['components']['policy'], 'policy')
        assert policy['type'] == policy_type
        assert policy['settings']['runtime']['quantization'] == quantization
        if node_path.startswith(('lora/', 'quic/')):
            assert configuration['checkpoint_from'] == node_path.split('/')[0]
    assert nodes['quic/full']['configuration']['evaluate']['checkpoint_from'] == 'quic'
    assert 'checkpoint_from' not in nodes['oft/full']['configuration']['evaluate']


@pytest.mark.parametrize('workers', [0, 33, True, '2'])
def test_invalid_yaml_scheduling_is_rejected(tmp_path, workers):
    document = _tree()
    document['execution']['max_workers'] = workers
    source = tmp_path / 'tree.yaml'
    source.write_text(dumps(document))
    with pytest.raises(ExperimentSpecError, match='execution.max_workers'):
        ExperimentWorkflow(ROOT).plan(source)


def test_standalone_execution_has_experiment_metadata_and_scoped_worker(tmp_path):
    calls = []
    def worker(command, **kwargs):
        calls.append(command[1])
        instance = kwargs['env']['OVLAB_EXPERIMENT_ID']
        result = ({'run_path': str(tmp_path / 'runs' / instance / 'evaluate' / 'run-one')}
                  if command[1] == '_native-run' else {'status': 'completed'})
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'success', 'result': result}))
    result = ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(tmp_path)}, runner=worker).run('experiments/evaluation/mock-e2e-smoke.yaml')
    assert result['status'] == 'completed'
    assert calls == ['_native-run', 'report']
    metadata = tmp_path / 'runs' / result['experiment_id'] / '.experiment'
    assert {p.name for p in metadata.iterdir()} == {'spec.json', 'spec.yaml', 'plan.json', 'result.json'}
    plan = json.loads((metadata / 'plan.json').read_text())
    assert plan['steps'] == [{'node': '', 'phase': 'evaluate'}]
    assert json.loads((metadata / 'spec.json').read_text())['kind'] == 'experiment'


@pytest.fixture
def stored_experiment(tmp_path):
    instance = 'study_2026-11-11_12-00-00_24265044c1'
    root = tmp_path / 'runs' / instance
    parent = root / 'evaluate' / 'model' / 'task'
    store = FilesystemRunArtifactStore(parent)
    runner = ExperimentRunner(
        runner_plan(), TrackingBenchmark(maximum_steps=3), TrackingPolicy(), store,
        clock=DeterministicClock(),
        configuration_snapshot=RunConfigurationSnapshot(
            'schema_version: "0.1.0"\nkind: scientific_experiment\n',
            f'scientific_config_hash: {"a" * 64}\nexecution_config_hash: {"b" * 64}\n',
            'a' * 64, 'b' * 64),
    )
    runner.connect()
    runner.run()
    run = store._run_path(runner.plan.run_context.run_id)
    spec = {'schema_version': 'ovlab.experiment/v1', 'kind': 'workflow', 'id': 'study', 'name': 'Study',
            'children': [{'id': 'model', 'children': [{'id': 'task', 'execute': ['evaluate'],
                         'config': {'evaluate': {'profile': 'mock-e2e-smoke.yaml'}}}]}]}
    source = tmp_path / 'source.yaml'
    source.write_text(dumps(spec))
    plan = ExperimentWorkflow(ROOT).plan(source)
    metadata = root / '.experiment'
    metadata.mkdir()
    documents = {'spec': spec, 'plan': {**plan, 'experiment_id': instance},
                 'result': {'schema_version': 'ovlab.experiment-result/v1', 'experiment_id': instance,
                            'spec_hash': _spec_hash(spec), 'status': 'completed',
                            'nodes': {'model/task': {'evaluate': {'run_id': run.name}}}}}
    for name, document in documents.items():
        (metadata / f'{name}.json').write_text(json.dumps(document))
    return tmp_path, root, run


def test_verify_instance_by_full_id_hash_or_path_is_read_only(stored_experiment):
    data, root, run = stored_experiment
    before = _hash_tree(data / 'runs')
    workflow = ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(data)})
    for reference in (root.name, '24265044c1', root):
        result = workflow.verify(reference)
        assert result['status'] == 'verified' and result['source_type'] == 'experiment'
        assert result['experiment_id'] == root.name
        assert len(result['verified_runs']) == 1
        assert result['verified_runs'][0]['integrity'] == 'verified'
    assert _hash_tree(data / 'runs') == before
    assert workflow.verify(str(run))['source_type'] == 'run'


def test_verify_instance_detects_tampered_canonical_evidence(stored_experiment):
    data, root, run = stored_experiment
    array = next(run.glob('tasks/*/episodes/*/arrays/*.npy'))
    array.write_bytes(array.read_bytes() + b'tampered')
    with pytest.raises(RunIntegrityError):
        ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(data)}).verify('24265044c1')


@pytest.mark.parametrize('damage', ['spec', 'missing-phase', 'extra-phase', 'symlink'])
def test_verify_rejects_inconsistent_experiment_metadata(stored_experiment, damage):
    data, root, run = stored_experiment
    metadata = root / '.experiment'
    if damage == 'spec':
        path = metadata / 'spec.json'
        document = json.loads(path.read_text()); document['name'] = 'changed'; path.write_text(json.dumps(document))
    elif damage == 'symlink':
        path = metadata / 'spec.json'
        real = metadata / 'other.json'; path.rename(real); path.symlink_to(real)
    else:
        path = metadata / 'result.json'
        document = json.loads(path.read_text())
        if damage == 'missing-phase':
            document['nodes'] = {}
        else:
            document['nodes']['extra'] = {'evaluate': {'run_id': 'extra'}}
        path.write_text(json.dumps(document))
    with pytest.raises(ExperimentIntegrityError):
        ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(data)}).verify('24265044c1')


def test_instance_hash_is_shared_with_automatic_reporting(stored_experiment):
    data, root, run = stored_experiment
    calls = []
    def worker(args, **kwargs):
        calls.append(kwargs['env']['OVLAB_EXPERIMENT_ID'])
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'success', 'result': {'output': '/exports/one'}}))
    result = generate_report('24265044c1', repository_root=ROOT, environment={'OVLAB_DATA_ROOT': str(data)}, runner=worker)
    assert result['source_type'] == 'experiment' and result['source'] == root.name
    assert calls == [root.name]


def test_instance_hash_rejects_collisions_and_symbolic_links(tmp_path):
    runs = tmp_path / 'runs'
    for name in ('a_24265044c1', 'b_24265044c1'):
        (runs / name / '.experiment').mkdir(parents=True)
    with pytest.raises(RunReferenceAmbiguousError):
        find_experiment(runs, '24265044c1')
    (runs / 'link').symlink_to(runs / 'a_24265044c1')
    with pytest.raises(RunReferenceError):
        find_experiment(runs, 'link')


def test_migrated_instances_verify_original_inventory_and_report_by_hash(stored_experiment):
    from ovlab_benchctl.experiment_migration import _inventory
    data, root, run = stored_experiment
    destination = root / 'evaluate' / run.name
    run.rename(destination)
    metadata = root / '.experiment'
    migration = {'schema_version': 'ovlab.experiment-migration/v1',
                 'source_inventory': _inventory(destination)}
    spec = {'schema_version': 'ovlab.experiment/v1', 'kind': 'workflow', 'id': root.name,
            'name': 'Migrated evaluation', 'execute': []}
    plan = {'schema_version': 'ovlab.experiment-plan/v1', 'experiment_id': root.name,
            'nodes': [{'id': root.name, 'path': ''}], 'steps': [], 'migration_only': True}
    result = {'schema_version': 'ovlab.experiment-result/v1', 'experiment_id': root.name,
              'status': 'migrated', 'migration': migration,
              'nodes': {'': {'evaluate': {'run_id': run.name, 'status': 'completed'}}}}
    for name, document in {'spec': spec, 'plan': plan, 'result': result, 'migration': migration}.items():
        (metadata / f'{name}.json').write_text(json.dumps(document))
    workflow = ExperimentWorkflow(ROOT, environment={'OVLAB_DATA_ROOT': str(data)})
    assert workflow.verify('24265044c1')['status'] == 'verified'
    def worker(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'success', 'result': {'output': '/exports/one'}}))
    reported = generate_report('24265044c1', repository_root=ROOT, environment=workflow.environment, runner=worker)
    assert reported['source_type'] == 'experiment' and len(reported['exports']) == 1
    migration['source_inventory']['sha256'] = '0' * 64
    result['migration'] = migration
    (metadata / 'migration.json').write_text(json.dumps(migration))
    (metadata / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ExperimentIntegrityError, match='original inventory'):
        workflow.verify('24265044c1')
