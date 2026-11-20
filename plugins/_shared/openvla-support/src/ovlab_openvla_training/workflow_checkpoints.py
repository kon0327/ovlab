"""Verified handoff from workflow training evidence to evaluation resources."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ovlab_core.experiment_scope import ExperimentScope

from ovlab_policy_sdk.training_errors import CheckpointResolutionError
from ovlab_policy_sdk.plugin import PolicyPluginCompatibilityError as ConfigCompatibilityError
from ovlab_policy_sdk.training_services import host_services
from .checkpoint_preparation import (
    CONVERTER_VERSION, IMAGE_CAPABILITY, IMAGE_CAPABILITY_LABEL,
)
from ovlab_policy_sdk.training_identity import atomic_json, canonical_json, sha256_file


def validate_checkpoint_binding(training, policy):
    mode = training['training']['mode']
    method = training['training'].get('peft', {}).get('method') if mode == 'peft' else 'full'
    expected = {'lora': 'openvla_lora_merged', 'quic': 'openvla_quic'}
    if method not in expected or policy.get('type') != expected[method]:
        raise ConfigCompatibilityError(
            f'checkpoint_from training method {method!r} is incompatible with evaluation policy {policy.get("type")!r}'
        )
    if method == 'lora' and policy['settings']['unnorm_key'] != 'libero_10':
        raise ConfigCompatibilityError('merged workflow LoRA currently requires LIBERO-10 normalization')
    return method


def _data_root(repository_root, environment):
    return Path(environment.get('OVLAB_MODEL_DATA_ROOT', environment.get(
        'OVLAB_DATA_ROOT', repository_root.parent / 'ovlab-data'))).expanduser().resolve()


def _files(snapshot):
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise CheckpointResolutionError('inference snapshot is missing or a symbolic link')
    for path in snapshot.rglob('*'):
        if path.is_symlink() or not path.is_file():
            raise CheckpointResolutionError('inference snapshots must contain only regular top-level files')
    names = {p.name for p in snapshot.iterdir()}
    if not {'config.json', 'model.safetensors.index.json', 'preprocessor_config.json', 'dataset_statistics.json'} <= names or not any(name.endswith('.safetensors') for name in names):
        raise CheckpointResolutionError('inference snapshot lacks complete model, processor, or normalization files')
    return names


@contextmanager
def _lock(path):
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _converter_image(environment, runner):
    image = environment.get('OVLAB_OPENVLA_IMAGE', 'ovlab-policy-openvla:local')
    result = runner(['docker', 'image', 'inspect', image, '--format',
                     '{{.Id}}|{{ index .Config.Labels "' + IMAGE_CAPABILITY_LABEL + '" }}'],
                    capture_output=True, text=True, check=False)
    if result.returncode:
        raise CheckpointResolutionError(f'LoRA preparation image is unavailable: {image}')
    digest, _, capability = result.stdout.strip().partition('|')
    if not digest or capability != IMAGE_CAPABILITY:
        raise CheckpointResolutionError(
            'LoRA checkpoint preparation requires an updated image; run bash deploy/scripts/build-images.sh policy-openvla'
        )
    return image, digest


def prepare_dense_checkpoint(resolved, checkpoint, model_data_root, environment, *, precision='bf16', runner=subprocess.run):
    """Publish one immutable, reusable snapshot; quantization happens later at model load."""
    method = checkpoint.get('planned_peft', {}).get('method') if checkpoint['kind'] == 'peft_adapter' else 'full'
    if method == 'lora':
        image, image_digest = _converter_image(environment, runner)
    else:
        raise CheckpointResolutionError(f'cannot prepare dense inference for {method!r}')
    if precision not in {'bf16', 'fp32'}:
        raise CheckpointResolutionError('unsupported training precision for dense checkpoint preparation')
    provenance = {'converter': CONVERTER_VERSION, 'checkpoint_id': resolved.checkpoint_id, 'precision': precision,
                  'base_sha256': resolved.base.spec.expected_sha256, 'image_digest': image_digest,
                  'checkpoint_manifest_sha256': sha256_file(resolved.bundle_host_path / 'manifest.json')}
    key = hashlib.sha256(canonical_json(provenance)).hexdigest()
    parent = model_data_root / 'checkpoints/local/inference' / resolved.checkpoint_id
    current = model_data_root
    for part in parent.relative_to(model_data_root).parts:
        current = current / part
        if current.is_symlink():
            raise CheckpointResolutionError('prepared checkpoint cache contains a symbolic link')
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / key
    with _lock(parent / (key + '.lock')):
        if final.is_symlink():
            raise CheckpointResolutionError('prepared checkpoint is a symbolic link')
        if not final.exists():
            temporary = Path(tempfile.mkdtemp(prefix='.' + key + '.', dir=parent))
            temporary.chmod(0o2770)
            snapshot = temporary / 'snapshot'
            snapshot.mkdir(mode=0o2770)
            snapshot.chmod(0o2770)
            try:
                if method == 'lora':
                    groups = sorted({snapshot.stat().st_gid, resolved.base.host_path.stat().st_gid,
                                     resolved.bundle_host_path.stat().st_gid})
                    group_args = [arg for group in groups for arg in ('--group-add', str(group))]
                    command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
                               '--user', '10001:10001', *group_args,
                               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                               '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m,mode=1777',
                               '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1',
                               '--env', 'CUDA_VISIBLE_DEVICES=',
                               '--mount', f'type=bind,source={resolved.base.host_path},target=/checkpoints/base,readonly',
                               '--mount', f'type=bind,source={resolved.bundle_host_path},target=/checkpoints/trained,readonly',
                               '--mount', f'type=bind,source={snapshot},target=/prepared',
                               '--entrypoint', 'python', image,
                               '-m', 'ovlab_openvla_training.checkpoint_preparation', '--base', '/checkpoints/base',
                               '--checkpoint', '/checkpoints/trained', '--output', '/prepared',
                               '--dtype', 'bfloat16' if precision == 'bf16' else 'float32']
                    result = runner(command, stdout=subprocess.PIPE, stderr=None, text=True, check=False)
                    if result.returncode:
                        raise CheckpointResolutionError('offline LoRA merge failed; evaluation was not started')
                _files(snapshot)
                revision = key[:40]
                spec = host_services().checkpoint_spec_from_snapshot(resolved.checkpoint_id, 'ovlab-local/' + resolved.checkpoint_id, revision, snapshot)
                files = {name: {'size': size, 'sha256': digest} for name, size, digest in spec.files}
                atomic_json(temporary / 'manifest.json', {'schema_version': CONVERTER_VERSION, 'provenance': provenance,
                            'revision': revision, 'expected_sha256': spec.expected_sha256, 'files': files})
                for path in snapshot.iterdir():
                    if path.stat().st_uid == os.getuid():
                        path.chmod(0o444)
                    elif path.stat().st_mode & 0o222:
                        raise CheckpointResolutionError('checkpoint converter left mutable output files')
                snapshot.chmod(0o555)
                (temporary / 'manifest.json').chmod(0o444)
                temporary.chmod(0o555)
                os.replace(temporary, final)
            except BaseException:
                for path in (temporary, snapshot):
                    if path.exists():
                        path.chmod(0o770)
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        manifest = final / 'manifest.json'
        if manifest.is_symlink() or not manifest.is_file():
            raise CheckpointResolutionError('prepared checkpoint has no regular manifest')
        recorded = json.loads(manifest.read_text())
        if recorded.get('schema_version') != CONVERTER_VERSION or recorded.get('provenance') != provenance:
            raise CheckpointResolutionError('prepared checkpoint provenance differs from the requested training output')
        snapshot = final / 'snapshot'
        if _files(snapshot) != set(recorded['files']):
            raise CheckpointResolutionError('prepared checkpoint file inventory changed')
        spec = host_services().checkpoint_spec(resolved.checkpoint_id, 'ovlab-local/' + resolved.checkpoint_id, recorded['revision'],
                              recorded['expected_sha256'], tuple((name, item['size'], item['sha256'])
                              for name, item in sorted(recorded['files'].items())))
        host_services().verify_checkpoint(spec, snapshot)
    entry = {'repo_id': spec.repo_id, 'revision': spec.revision, 'expected_sha256': spec.expected_sha256,
             'files': recorded['files']}
    if method == 'lora':
        peft = checkpoint['planned_peft']
        entry['artifact'] = {'form': 'merged_full_weights', 'merge_status': 'merged', 'active_peft_adapter': False,
                             'runtime_peft_modules': False, 'adapter_config': 'not_present_in_published_artifact',
                             'adapter_recoverability': 'not_recoverable_from_published_artifact', 'files': recorded['files']}
        entry['method'] = {'id': 'ovlab-trained-lora', 'version': '1.0.0', 'family': 'lora',
                           'declared_base_model': resolved.base.spec.repo_id, 'declared_base_revision': resolved.base.spec.revision,
                           'adaptation_suite': 'LIBERO-10', 'quantization': 'none',
                           'lora': {'rank': peft['rank'], 'alpha': peft['alpha'], 'scaling': peft['alpha'] / peft['rank'],
                                    'dropout': peft['dropout'], 'bias': 'none', 'target_policy': 'all-linear',
                                    'modules_to_save': None, 'merge_procedure': 'merge_and_unload()+save_pretrained()'},
                           'training_provenance': {'origin': 'ovlab-training', 'checkpoint_id': resolved.checkpoint_id,
                                                   'training_run_id': resolved.training_run_id,
                                                   'conversion': provenance}}
    return snapshot, entry


def bind_training_checkpoint(repository_root, environment, configuration, record, *, runs_root, experiment_id,
                             bundle, expected_profile, runner=subprocess.run, prepare=None):
    """Use only the finalized checkpoint actually recorded by the declared source phase."""
    if not isinstance(record, dict) or not isinstance(record.get('run_id'), str):
        raise CheckpointResolutionError('checkpoint_from source training has no completed run')
    source = configuration['checkpoint_from']
    scope = ExperimentScope(experiment_id, tuple(source.split('/')) if source else ())
    source_root = scope.runs_root(runs_root, 'train')
    if Path(str(record.get('run_path', ''))).resolve() != (source_root / record['run_id']).resolve():
        raise CheckpointResolutionError('checkpoint_from source run is outside its declared experiment node')
    host_services().resolve_run_reference(runs_root, str(source_root / record['run_id']), category='train')
    data_root = _data_root(repository_root, environment)
    store = host_services().training_store(data_root, training_runs_root=source_root)
    store.verify(record['run_id'])
    training = store.inspect(record['run_id'])
    if training['result'].get('status') != 'completed':
        raise CheckpointResolutionError('checkpoint_from training is not completed')
    if training['plan']['scientific']['profile'] != expected_profile:
        raise CheckpointResolutionError('checkpoint_from training profile differs from the planned source')
    checkpoint_id = training['result'].get('checkpoint_id')
    if not isinstance(checkpoint_id, str):
        raise CheckpointResolutionError('checkpoint_from training did not finalize a checkpoint')
    reported_id = record.get('result', {}).get('result', {}).get('checkpoint_id')
    if reported_id is not None and reported_id != checkpoint_id:
        raise CheckpointResolutionError('checkpoint_from CLI result differs from canonical training evidence')
    config = deepcopy(configuration)
    policy = config.get('component_documents', {}).get('policy', {}).get('document')
    if policy is None:
        policy = bundle.load_component(config['document']['components']['policy'], 'policy')
    policy = deepcopy(policy)
    method = validate_checkpoint_binding(expected_profile, policy)
    if 'deployment' not in config['document']:
        raise ConfigCompatibilityError('checkpoint_from requires a deployed evaluation policy')
    registry = bundle.load_component(config['document']['resources']['registry'], 'resource_registry')
    base_id = training['plan']['scientific']['base_checkpoint']['resource_id']
    local_profile = environment.get('OVLAB_LOCAL_PROFILE')
    resolved = host_services().resolve_finalized_training_checkpoint(
        repository_root, data_root, checkpoint_id,
        global_cache=Path(environment.get('OVLAB_GLOBAL_HF_CACHE') or host_services().default_global_cache(environment)),
        local_base_path=host_services().local_checkpoint_override(local_profile, base_id),
    )
    checkpoint = host_services().checkpoint_store(data_root).inspect(checkpoint_id)['checkpoint']
    expected_loader = {'lora': 'ovlab-openvla-peft', 'quic': 'ovlab-openvla-quic'}[method]
    if resolved.expected_loader != expected_loader:
        raise CheckpointResolutionError('training output loader is incompatible with the evaluation policy')
    if checkpoint.get('planned_peft') != expected_profile['training']['peft']:
        raise CheckpointResolutionError('finalized adapter differs from the declared training method')
    proof = {'schema_version': 'ovlab.workflow-checkpoint-binding/v1', 'source_node': source,
             'source_run_id': record['run_id'], 'checkpoint_id': checkpoint_id, 'method': method,
             'training_profile_sha256': hashlib.sha256(canonical_json(expected_profile)).hexdigest(),
             'checkpoint_manifest_sha256': sha256_file(resolved.bundle_host_path / 'manifest.json'),
             'base_checkpoint': {'resource_id': resolved.base.spec.resource_id,
                                 'repository': resolved.base.spec.repo_id, 'revision': resolved.base.spec.revision,
                                 'aggregate_sha256': resolved.base.spec.expected_sha256}}
    overrides = {}
    if method == 'quic':
        adapter = resolved.bundle_host_path / 'weights-or-adapter'
        settings = policy['settings']['quic']
        artifact = json.loads((adapter / 'adapter.json').read_text())
        metadata = artifact.get('metadata', {})
        if metadata.get('normalization_contract', {}).get('unnorm_key') != policy['settings']['unnorm_key']:
            raise CheckpointResolutionError('trained QuIC normalization differs from evaluation settings')
        settings.update(adapter_path=str(adapter), adapter_manifest_sha256=sha256_file(adapter / 'adapter.json'),
                        adapter_tensor_sha256=sha256_file(adapter / 'adapter.safetensors'),
                        adaptation_state=artifact['host_identity']['adaptation_state'],
                        **metadata.get('preprocessing', {'prompt_style': 'modern', 'center_crop': False}))
        policy['settings']['checkpoint_id'] = policy['settings']['processor_id'] = base_id
        entry = registry['checkpoints'][base_id]
        entry.update(revision=resolved.base.spec.revision, expected_sha256=resolved.base.spec.expected_sha256,
                     files={name: {'size': size, 'sha256': digest} for name, size, digest in resolved.base.spec.files})
        overrides['OVLAB_QUIC_ADAPTER_HOST_PATH'] = str(adapter)
        overrides.update(OVLAB_WORKFLOW_CHECKPOINT_ID=base_id, OVLAB_WORKFLOW_CHECKPOINT_PATH=str(resolved.base.host_path))
        proof['adapter_manifest_sha256'] = settings['adapter_manifest_sha256']
        proof['adapter_tensor_sha256'] = settings['adapter_tensor_sha256']
    else:
        preparation_environment = dict(environment)
        if 'OVLAB_OPENVLA_IMAGE' not in preparation_environment:
            env_file = Path(environment.get('OVLAB_COMPOSE_ENV_FILE') or repository_root / 'deploy/compose/.env')
            if not env_file.is_absolute():
                env_file = repository_root / env_file
            if env_file.is_file():
                image = host_services().image_environment(repository_root, environment, str(env_file), 'OVLAB_OPENVLA_IMAGE')
                if image:
                    preparation_environment['OVLAB_OPENVLA_IMAGE'] = image
        snapshot, entry = (prepare or prepare_dense_checkpoint)(resolved, checkpoint, data_root, preparation_environment,
                                                   precision=expected_profile['training']['precision'], runner=runner)
        statistics = json.loads((snapshot / 'dataset_statistics.json').read_text())
        if policy['settings']['unnorm_key'] not in statistics:
            raise CheckpointResolutionError('trained checkpoint lacks the requested evaluation normalization')
        registry['checkpoints'][checkpoint_id] = entry
        policy['settings']['checkpoint_id'] = policy['settings']['processor_id'] = checkpoint_id
        overrides.update(OVLAB_WORKFLOW_CHECKPOINT_ID=checkpoint_id, OVLAB_WORKFLOW_CHECKPOINT_PATH=str(snapshot))
        proof.update(inference_sha256=entry['expected_sha256'], inference_resource=checkpoint_id)
    host_services().validate_document(policy, 'workflow-bound-policy', 'policy')
    host_services().validate_document(registry, 'workflow-bound-registry', 'resource_registry')
    selected_policy = config.setdefault('component_documents', {}).setdefault('policy', {})
    selected_policy.update(document=policy, profile=config['document']['components']['policy'])
    config['registry_document'] = registry
    proof['configuration_sha256'] = hashlib.sha256(canonical_json(config)).hexdigest()
    return config, proof, overrides


