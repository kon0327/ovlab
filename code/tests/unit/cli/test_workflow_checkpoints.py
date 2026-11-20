"""Training-to-evaluation handoff must never fall back to reference weights."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import struct
from types import SimpleNamespace

import pytest

from ovlab_core.experiment_scope import ExperimentScope
from ovlab_benchctl.checkpointing import checkpoint_spec_from_snapshot, CheckpointResolutionError
from ovlab_benchctl.checkpoint_preparation import finish_snapshot
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.errors import ConfigCompatibilityError
from ovlab_benchctl.experiment_workflow import ExperimentSpecError, ExperimentWorkflow
from ovlab_benchctl.strict_yaml import dumps, load
from ovlab_benchctl.training_composition import load_training_profile
from ovlab_benchctl.training_identity import atomic_json, inventory, sha256_file
from ovlab_benchctl.training_runs import CheckpointBundleStore, TrainingRunStore
from ovlab_benchctl import workflow_checkpoints as handoff
from ovlab_openvla_common import OpenVlaRuntimeArtifact, method_descriptor_from_registry

ROOT = Path(__file__).resolve().parents[4]


def tensor(path, name='layer.weight', value=1.0):
    raw = struct.pack('<f', value)
    header = json.dumps({name: {'dtype': 'F32', 'shape': [1, 1], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + raw)


@pytest.fixture
def resources(tmp_path):
    repository, data = tmp_path / 'repository', tmp_path / 'data'
    shutil.copytree(ROOT / 'configs', repository / 'configs')
    shutil.copytree(ROOT / 'experiments', repository / 'experiments')
    registrations = json.loads((ROOT / 'plugins/registry.json').read_text())
    registrations['plugins']['openvla']['source']['path'] = str(ROOT / 'plugins/openvla')
    (repository / 'plugins').mkdir()
    (repository / 'plugins/registry.json').write_text(json.dumps(registrations))
    # This fixture supplies a tiny, explicit resource catalog; real plugin
    # defaults remain read-only and still provide the policy implementations.
    for experiment in (repository / 'experiments/evaluation').glob('*.yaml'):
        experiment.write_text(experiment.read_text().replace(
            'plugin://openvla/resources/registry.yaml', 'resources/registry.yaml'))

    revision = 'c' * 40
    base = data / 'checkpoints/huggingface/openvla-7b' / revision
    base.mkdir(parents=True)
    (base / 'config.json').write_text(json.dumps({'model_type': 'openvla', 'norm_stats': {'bridge_orig': {}}}))
    (base / 'preprocessor_config.json').write_text('{}')
    tensor(base / 'model.safetensors')
    spec = checkpoint_spec_from_snapshot('openvla-7b', 'tiny/openvla', revision, base)
    registry = load(ROOT / 'plugins/openvla/configs/resources/registry.yaml')
    registry['checkpoints']['openvla-7b'] = {
        'repo_id': spec.repo_id, 'revision': revision, 'expected_sha256': spec.expected_sha256,
        'files': {name: {'size': size, 'sha256': digest} for name, size, digest in spec.files},
    }
    (repository / 'configs/resources/registry.yaml').write_text(dumps(registry))
    profile = deepcopy(load_training_profile(repository / 'experiments/training/openvla-lora-libero10-smoke.yaml').document)
    profile['model']['revision'] = revision
    profile['training']['peft'].update(rank=2, alpha=2)
    return SimpleNamespace(repository=repository, data=data, base=base, spec=spec, profile=profile,
                           environment={'OVLAB_DATA_ROOT': str(data), 'OVLAB_GLOBAL_HF_CACHE': str(tmp_path / 'empty-cache')})


def trained(resources, *, instance='study', node='lora', method='lora', finalized=True):
    profile = deepcopy(resources.profile)
    if method == 'quic':
        profile['training']['peft'] = {'method': 'quic', 'placement': {'id': 'custom'},
                                    'adapter': {'profile': 'QP2'}, 'targets': [{'names': ['layer']} ]}
        profile['trainer']['implementation'] = 'openvla-quic'
    base = {'resource_id': resources.spec.resource_id, 'repository': resources.spec.repo_id,
            'revision': resources.spec.revision, 'aggregate_sha256': resources.spec.expected_sha256}
    plan = {'profile_id': profile['id'], 'scientific_training_id': 'training-plan-' + 'a' * 32,
            'execution_plan_id': 'training-execution-' + 'b' * 32,
            'scientific': {'profile': profile, 'training': profile['training'],
                           'output': {'output_kind': 'adapter'}, 'base_checkpoint': base}}
    store = TrainingRunStore(resources.data, scope=ExperimentScope(instance, tuple(node.split('/')) if node else ()))
    context = store.create(profile, profile, plan, {})
    staging = context.staging_checkpoint
    for directory in ('weights-or-adapter', 'processor', 'auxiliary'):
        (staging / directory).mkdir(parents=True)
    (staging / 'processor/preprocessor_config.json').write_text('{}')
    (staging / 'auxiliary/dataset_statistics.json').write_text(json.dumps({'libero_10': {'action': {'q01': [-.5]*7, 'q99': [.5]*7}}}))
    weights = staging / 'weights-or-adapter'
    if method == 'lora':
        (weights / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'r': 2, 'lora_alpha': 2, 'target_modules': ['all-linear']}))
        tensor(weights / 'adapter_model.safetensors', name='layer.lora_A.weight', value=2.0)
        if finalized:
            CheckpointBundleStore(resources.data).finalize(context)
    else:
        # A finalized fixture exercises the same immutable manifests and loader
        # checks without running the separate QuIC optimizer/target resolver.
        tensor(weights / 'adapter.safetensors', name='layer.adapter.entries', value=2.0)
        payload = {'format_version': 1, 'host_identity': {'adaptation_state': 'none', 'revision': base['revision']},
                   'metadata': {'normalization_contract': {'unnorm_key': 'libero_10'},
                                'preprocessing': {'prompt_style': 'modern', 'center_crop': False}},
                   'tensor_sha256': sha256_file(weights / 'adapter.safetensors')}
        (weights / 'adapter.json').write_text(json.dumps(payload))
        checkpoint_id = 'checkpoint-' + 'd' * 32
        destination = resources.data / 'checkpoints/local' / checkpoint_id
        shutil.copytree(staging, destination)
        document = {'checkpoint_id': checkpoint_id, 'kind': 'peft_adapter', 'merge_status': 'unmerged',
                    'base_checkpoint': base, 'expected_loader': 'ovlab-openvla-quic',
                    'planned_peft': profile['training']['peft'], 'training_run_id': context.run_id}
        atomic_json(destination / 'checkpoint.json', document)
        files, _, _ = inventory(destination)
        atomic_json(destination / 'manifest.json', {'schema_version': 'ovlab.checkpoint-manifest/v1',
                    'state': 'finalized', 'checkpoint_id': checkpoint_id, 'files': files})
        CheckpointBundleStore._complete_run(context, {}, checkpoint_id, destination)
    record = {'run_id': context.run_id, 'run_path': str(context.root), 'result': store.inspect(context.run_id)}
    return profile, record


def evaluation(resources, *, method='lora', source='lora'):
    filename = 'libero10-quic-adapter-rpc-smoke.yaml' if method == 'quic' else 'libero10-lora-full-three-per-task.yaml'
    plan = ExperimentWorkflow(resources.repository).plan(resources.repository / 'experiments/evaluation' / filename)
    configuration = deepcopy(plan['nodes'][0]['configuration']['evaluate'])
    configuration['checkpoint_from'] = source
    return configuration


class Converter:
    def __init__(self, *, fail=False):
        self.merges, self.fail = 0, fail
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[:3] == ['docker', 'image', 'inspect']:
            return SimpleNamespace(returncode=0, stdout='sha256:image|lora-merge-v1')
        self.merges += 1
        if self.fail:
            return SimpleNamespace(returncode=1, stdout='')
        mounts = [command[index+1] for index, arg in enumerate(command) if arg == '--mount']
        paths = {}
        for mount in mounts:
            fields = dict(part.split('=', 1) for part in mount.split(',') if '=' in part)
            paths[fields['target']] = Path(fields['source'])
        output, bundle = paths['/prepared'], paths['/checkpoints/trained']
        (output / 'config.json').write_text(json.dumps({'model_type': 'openvla'}))
        tensor(output / 'model.safetensors', value=3.0)
        finish_snapshot(output, bundle)
        return SimpleNamespace(returncode=0, stdout='{}')


def bind(resources, configuration, profile, record, converter):
    return handoff.bind_training_checkpoint(resources.repository, resources.environment, configuration, record,
        runs_root=resources.data / 'runs', experiment_id='study',
        bundle=ConfigBundleBuilder(resources.repository), expected_profile=profile, runner=converter)


def test_lora_uses_new_checkpoint_and_reuses_one_dense_snapshot_for_all_precisions(resources):
    profile, record = trained(resources)
    converter = Converter()
    configuration = evaluation(resources)
    original = deepcopy(configuration)
    outputs = []
    for precision in ('none', '8bit', '4bit'):
        cfg = deepcopy(configuration)
        policy = ConfigBundleBuilder(resources.repository).load_component(cfg['document']['components']['policy'], 'policy')
        policy['settings']['runtime']['quantization'] = precision
        cfg['component_documents']['policy'] = {'document': policy}
        bound, proof, env = bind(resources, cfg, profile, record, converter)
        checkpoint_id = record['result']['result']['checkpoint_id']
        selected = bound['component_documents']['policy']['document']['settings']
        assert selected['checkpoint_id'] == selected['processor_id'] == checkpoint_id
        assert selected['runtime']['quantization'] == precision
        assert proof['checkpoint_id'] == checkpoint_id and proof['source_run_id'] == record['run_id']
        snapshot = Path(env['OVLAB_WORKFLOW_CHECKPOINT_PATH'])
        assert env['OVLAB_WORKFLOW_CHECKPOINT_ID'] == checkpoint_id
        entry = bound['registry_document']['checkpoints'][checkpoint_id]
        OpenVlaRuntimeArtifact.from_registry_entry(checkpoint_id, entry).verify(snapshot)
        descriptor = method_descriptor_from_registry(entry)
        assert descriptor.lora_configuration['rank'] == 2
        assert descriptor.training_provenance['checkpoint_id'] == checkpoint_id
        assert json.loads((snapshot / 'config.json').read_text())['norm_stats']['libero_10']
        outputs.append(snapshot)
    assert converter.merges == 1 and len(set(outputs)) == 1
    assert configuration == original
    assert '--network=none' in converter.commands[1] and '--read-only' in converter.commands[1]
    assert '--gpus' not in converter.commands[1]
    assert not (resources.base / 'dataset_statistics.json').exists()


def test_quic_uses_fresh_adapter_hashes_base_and_normalization(resources):
    profile, record = trained(resources, node='quic', method='quic')
    configuration = evaluation(resources, method='quic', source='quic')
    bound, proof, env = bind(resources, configuration, profile, record, Converter(fail=True))
    settings = bound['component_documents']['policy']['document']['settings']
    adapter = Path(env['OVLAB_QUIC_ADAPTER_HOST_PATH'])
    assert settings['checkpoint_id'] == settings['processor_id'] == 'openvla-7b'
    assert settings['quic']['adapter_manifest_sha256'] == sha256_file(adapter / 'adapter.json')
    assert settings['quic']['adapter_tensor_sha256'] == sha256_file(adapter / 'adapter.safetensors')
    assert settings['quic']['adaptation_state'] == 'none'
    assert proof['checkpoint_id'] == record['result']['result']['checkpoint_id']
    assert Path(env['OVLAB_WORKFLOW_CHECKPOINT_PATH']) == resources.base


def test_lora_preparation_honors_compose_policy_image_override(resources):
    env_file = resources.repository / 'deploy/compose/.env'
    env_file.parent.mkdir(parents=True)
    env_file.write_text('OVLAB_OPENVLA_IMAGE=custom-policy:trained\n')
    profile, record = trained(resources)
    converter = Converter()
    bind(resources, evaluation(resources), profile, record, converter)
    assert converter.commands[0][3] == 'custom-policy:trained'


@pytest.mark.parametrize('failure', ['unfinalized', 'wrong-source', 'wrong-profile', 'wrong-result', 'conversion'])
def test_invalid_training_handoff_never_reaches_reference_evaluation(resources, failure):
    profile, record = trained(resources, finalized=failure != 'unfinalized')
    configuration = evaluation(resources)
    converter = Converter(fail=failure == 'conversion')
    if failure == 'wrong-source':
        record['run_path'] = str(resources.data / 'runs/study/train/other' / record['run_id'])
    elif failure == 'wrong-profile':
        profile['training']['seed'] += 1
    elif failure == 'wrong-result':
        record['result']['result']['checkpoint_id'] = 'checkpoint-' + '0' * 32
    with pytest.raises((RuntimeError, ValueError)):
        bind(resources, configuration, profile, record, converter)
    assert configuration['document']['components']['policy'] == 'plugin://openvla/policies/openvla-lora/merged-libero10.yaml'
    if failure == 'conversion':
        assert not list((resources.data / 'checkpoints/local/inference').rglob('manifest.json'))
    else:
        assert converter.merges == 0


def test_cached_snapshot_tampering_fails_without_remerging(resources):
    profile, record = trained(resources)
    configuration = evaluation(resources)
    converter = Converter()
    _, _, env = bind(resources, configuration, profile, record, converter)
    weight = Path(env['OVLAB_WORKFLOW_CHECKPOINT_PATH']) / 'model.safetensors'
    weight.chmod(0o644)
    weight.write_bytes(weight.read_bytes() + b'tamper')
    with pytest.raises(CheckpointResolutionError, match='size mismatch'):
        bind(resources, configuration, profile, record, converter)
    assert converter.merges == 1


def workflow_document():
    return {'schema_version': 'ovlab.experiment/v1', 'kind': 'workflow', 'id': 'study', 'name': 'Study',
            'config': {'train': {'profile': 'openvla-lora-libero10-smoke.yaml'}},
            'children': [{'id': 'lora', 'execute': ['train'], 'config': {'evaluate': {
                'checkpoint_from': 'lora', 'profile': 'libero10-lora-full-three-per-task.yaml'}},
                'children': [{'id': 'full', 'execute': ['evaluate']}, {'id': 'eight', 'execute': ['evaluate'],
                             'config': {'evaluate': {'component_patches': {'policy': {'settings': {'runtime': {'quantization': '8bit'}}}}}}}]}]}


def test_checkpoint_from_is_inherited_and_adds_the_training_dependency(tmp_path):
    source = tmp_path / 'tree.yaml'
    source.write_text(dumps(workflow_document()))
    plan = ExperimentWorkflow(ROOT).plan(source)
    assert plan['steps'][0] == {'node': 'lora', 'phase': 'train'}
    for node in plan['nodes'][2:]:
        assert node['configuration']['evaluate']['checkpoint_from'] == 'lora'
        assert node['depends_on'] == []
    with pytest.raises(ExperimentSpecError, match='missing training phase'):
        ExperimentWorkflow(ROOT).plan(source, node='lora/full')


@pytest.mark.parametrize('change', ['missing', 'cycle', 'wrong-method'])
def test_invalid_checkpoint_dependencies_are_rejected_before_workloads(tmp_path, change):
    document = workflow_document()
    if change == 'missing':
        document['children'][0]['config']['evaluate']['checkpoint_from'] = 'unknown'
    elif change == 'cycle':
        document['children'][0]['depends_on'] = [{'node': 'lora/full', 'phase': 'evaluate'}]
    else:
        document['children'][0]['config']['evaluate']['profile'] = 'libero10-oft-full-three-per-task.yaml'
    source = tmp_path / 'tree.yaml'
    source.write_text(dumps(document))
    with pytest.raises((ExperimentSpecError, ConfigCompatibilityError)):
        ExperimentWorkflow(ROOT).plan(source)


def run_fixture_workflow(resources, monkeypatch, *, finalized=True):
    from helpers.runner_fixtures import TrackingBenchmark, TrackingPolicy, runner_plan
    from ovlab_runner import DeterministicClock, ExperimentRunner, FilesystemRunArtifactStore, RunConfigurationSnapshot
    converter = Converter()
    original_prepare = handoff.prepare_dense_checkpoint
    def prepare(*args, **kwargs):
        kwargs['runner'] = converter
        return original_prepare(*args, **kwargs)
    monkeypatch.setattr(handoff, 'prepare_dense_checkpoint', prepare)
    seen = []
    source = resources.repository / 'experiment.yaml'
    document = workflow_document()
    document['execution'] = {'max_workers': 2}
    document['config']['train']['patch'] = {'model': {'revision': resources.spec.revision},
                                          'training': {'peft': {'rank': 2, 'alpha': 2}}}
    source.write_text(dumps(document))
    def worker(command, **kwargs):
        environment = kwargs['env']
        instance, node = environment['OVLAB_EXPERIMENT_ID'], environment['OVLAB_EXPERIMENT_NODE']
        seen.append((command[1], node))
        if command[1] == 'train':
            _, record = trained(resources, instance=instance, node=node, finalized=finalized)
            result = record['result']
        elif command[1] == 'eval':
            builder = ConfigBundleBuilder(resources.repository)
            generated = Path(command[3]).relative_to(resources.repository).as_posix()
            experiment = builder.load_component(generated, 'experiment')
            policy = builder.load_component(experiment['components']['policy'], 'policy')
            checkpoint_id = environment['OVLAB_WORKFLOW_CHECKPOINT_ID']
            assert policy['settings']['checkpoint_id'] == checkpoint_id
            assert checkpoint_id.startswith('checkpoint-')
            registry = builder.load_component(experiment['resources']['registry'], 'resource_registry')
            assert registry['checkpoints'][checkpoint_id]['method']['training_provenance']['checkpoint_id'] == checkpoint_id
            bundle = builder.build(generated, None)
            parent = ExperimentScope(instance, tuple(node.split('/'))).runs_root(resources.data / 'runs', 'evaluate')
            artifacts = FilesystemRunArtifactStore(parent)
            runner = ExperimentRunner(runner_plan(), TrackingBenchmark(maximum_steps=2), TrackingPolicy(), artifacts,
                clock=DeterministicClock(), configuration_snapshot=RunConfigurationSnapshot(
                    'schema_version: "0.1.0"\nkind: scientific_experiment\n',
                    'scientific_config_hash: ' + 'a'*64 + '\nexecution_config_hash: ' + 'b'*64 + '\n', 'a'*64, 'b'*64))
            runner.connect()
            runner.run()
            result = {'run_path': str(artifacts._run_path(runner.plan.run_context.run_id)), 'config_bundle': bundle.summary()}
        else:
            result = {'output': str(resources.data / 'exports' / instance / node), 'status': 'completed'}
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'success', 'result': result}))
    return ExperimentWorkflow(resources.repository, environment=resources.environment, runner=worker), source, seen, converter


def test_scheduler_persists_and_verifies_source_binding_for_parallel_evaluations(resources, monkeypatch):
    from ovlab_benchctl.experiment_workflow import ExperimentIntegrityError
    workflow, source, seen, converter = run_fixture_workflow(resources, monkeypatch)
    result = workflow.run(source)
    assert result['status'] == 'completed'
    assert converter.merges == 1
    training = result['nodes']['lora']['train']
    for path in ('lora/full', 'lora/eight'):
        binding = result['nodes'][path]['evaluate']['checkpoint_binding']
        assert binding['source_run_id'] == training['run_id']
        assert binding['checkpoint_id'] == training['result']['result']['checkpoint_id']
    assert workflow.verify(result['experiment_id'])['status'] == 'verified'
    root = resources.data / 'runs' / result['experiment_id'] / '.experiment'
    binding_path = root / 'checkpoint-bindings/nodes/lora/full.json'
    saved = json.loads(binding_path.read_text())
    assert saved['configuration']['checkpoint_from'] == 'lora'
    assert not list((resources.repository / 'configs/resources').glob('.generated/*'))
    result['nodes']['lora/full']['evaluate']['checkpoint_binding']['checkpoint_id'] = 'checkpoint-' + '0'*32
    (root / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ExperimentIntegrityError, match='binding'):
        workflow.verify(result['experiment_id'])


def test_unfinalized_training_result_stops_workflow_before_evaluation(resources, monkeypatch):
    workflow, source, seen, converter = run_fixture_workflow(resources, monkeypatch, finalized=False)
    with pytest.raises(RuntimeError, match='not finalized'):
        workflow.run(source)
    assert not [call for call in seen if call[0] == 'eval']
    assert converter.merges == 0


def test_reused_immutable_checkpoint_still_binds_the_current_training_run(resources):
    first_profile, first = trained(resources)
    converter = Converter()
    configuration = evaluation(resources)
    bind(resources, configuration, first_profile, first, converter)
    second_profile, second = trained(resources)
    assert first['run_id'] != second['run_id']
    assert first['result']['result']['checkpoint_id'] == second['result']['result']['checkpoint_id']
    _, proof, _ = bind(resources, configuration, second_profile, second, converter)
    assert proof['source_run_id'] == second['run_id']
    assert converter.merges == 1


def test_bound_snapshot_is_verified_without_huggingface_fallback(resources):
    from ovlab_benchctl.deployment import ComposeDeployment, ComposeDeploymentError
    profile, record = trained(resources)
    configuration = evaluation(resources)
    bound, _, env = bind(resources, configuration, profile, record, Converter())
    workflow = ExperimentWorkflow(resources.repository, environment=resources.environment)
    generated = workflow._materialize('study', 'evaluation', 'evaluate', bound)
    env_file = resources.repository / 'deploy/compose/.env'
    env_file.parent.mkdir(parents=True)
    env_file.write_text('')
    deployment = ComposeDeployment(resources.repository, environment=env)
    plan = deployment.plan(generated)
    selected = deployment._resolve_checkpoint(plan)
    assert selected.spec.resource_id == record['result']['result']['checkpoint_id']
    assert selected.source_kind == 'workflow-training-output'
    deployment.environment['OVLAB_WORKFLOW_CHECKPOINT_ID'] = 'wrong'
    with pytest.raises(ComposeDeploymentError, match='differs'):
        deployment._resolve_checkpoint(plan)
