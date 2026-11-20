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

from .checkpointing import (
    CheckpointResolutionError, CheckpointSpec, checkpoint_spec_from_snapshot,
    default_global_cache, resolve_finalized_training_checkpoint, verify_checkpoint,
)
from .errors import ConfigCompatibilityError
from .schema import validate
from .training_identity import atomic_json, canonical_json, sha256_file
from .training_runs import CheckpointBundleStore, TrainingRunStore


def validate_checkpoint_binding(training, policy):
    from .training_plugins import backend_for_profile
    return backend_for_profile(training).validate_checkpoint_binding(training, policy)


def prepare_dense_checkpoint(resolved, checkpoint, model_data_root, environment, **options):
    from .training_plugins import backend_for_checkpoint
    return backend_for_checkpoint(checkpoint).prepare_dense_checkpoint(
        resolved, checkpoint, model_data_root, environment, **options)


def bind_training_checkpoint(repository_root, environment, configuration, record, *, runs_root, experiment_id,
                             bundle, expected_profile, runner=subprocess.run):
    from .training_plugins import backend_for_profile
    return backend_for_profile(expected_profile, repository_root).bind_training_checkpoint(
        repository_root, environment, configuration, record, runs_root=runs_root,
        experiment_id=experiment_id, bundle=bundle, expected_profile=expected_profile,
        runner=runner, prepare=prepare_dense_checkpoint)


def _data_root(repository_root, environment):
    return Path(environment.get('OVLAB_MODEL_DATA_ROOT', environment.get(
        'OVLAB_DATA_ROOT', repository_root.parent / 'ovlab-data'))).expanduser().resolve()


def verify_saved_binding(repository_root, environment, configured, proof, nodes, runs_root,
                         experiment_id, binding_file, verified_checkpoints, planned_training_profile):
    """Check persisted handoff identity against canonical training and checkpoint evidence."""
    if not isinstance(proof, dict) or proof.get('source_node') != configured['checkpoint_from']:
        raise CheckpointResolutionError('missing or incorrect source checkpoint binding')
    current = runs_root
    for part in binding_file.relative_to(runs_root).parts:
        current = current / part
        if current.is_symlink():
            raise CheckpointResolutionError('checkpoint binding snapshot contains a symbolic link')
    if not binding_file.is_file():
        raise CheckpointResolutionError('checkpoint binding snapshot is unavailable')
    saved = json.loads(binding_file.read_text())
    configuration = saved.get('configuration')
    if (not isinstance(configuration, dict) or configuration.get('checkpoint_from') != configured['checkpoint_from'] or
            saved.get('binding') != proof or hashlib.sha256(canonical_json(configuration)).hexdigest() != proof.get('configuration_sha256')):
        raise CheckpointResolutionError('bound configuration snapshot differs from its recorded identity')
    source = nodes[proof['source_node']]['train']
    if source['run_id'] != proof.get('source_run_id'):
        raise CheckpointResolutionError('evaluation refers to another training run')
    scope = ExperimentScope(experiment_id, tuple(proof['source_node'].split('/')) if proof['source_node'] else ())
    data_root = _data_root(repository_root, environment)
    from .run_references import resolve_run_reference_tree
    resolve_run_reference_tree(runs_root, str(scope.runs_root(runs_root, 'train') / source['run_id']), category='train')
    training = TrainingRunStore(data_root, training_runs_root=scope.runs_root(runs_root, 'train')).inspect(source['run_id'])
    profile = training['plan']['scientific']['profile']
    if profile != planned_training_profile or hashlib.sha256(canonical_json(profile)).hexdigest() != proof.get('training_profile_sha256'):
        raise CheckpointResolutionError('source training profile differs from the experiment binding')
    if training['result'].get('status') != 'completed' or training['result'].get('checkpoint_id') != proof.get('checkpoint_id'):
        raise CheckpointResolutionError('evaluation checkpoint differs from the source training result')
    store = CheckpointBundleStore(data_root)
    if proof['checkpoint_id'] not in verified_checkpoints:
        store.verify(proof['checkpoint_id'])
        verified_checkpoints.add(proof['checkpoint_id'])
    inspection = store.inspect(proof['checkpoint_id'])
    if sha256_file(Path(inspection['host_path']) / 'manifest.json') != proof.get('checkpoint_manifest_sha256'):
        raise CheckpointResolutionError('source checkpoint manifest identity changed')
