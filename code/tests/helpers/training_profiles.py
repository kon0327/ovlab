"""Small test recipes composed from the public model-plugin defaults.

These are fixtures for behavior tests, not an archived production preset catalog.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from ovlab_benchctl.strict_yaml import dumps, load

ROOT = Path(__file__).resolve().parents[3]


def training_document(method='lora'):
    project = 'openvla-quic' if method == 'quic' else 'openvla'
    document = deepcopy(load(ROOT / 'plugins' / project / 'configs/train/default.yaml'))
    document['id'] = f'test-{method}'
    document['dataset']['ref'] = 'libero/libero_10'
    document['training'].update(max_steps=3, per_device_batch_size=1, gradient_accumulation_steps=1)
    document['resources']['max_vram_gib'] = 32 if method != 'full' else 80
    if method == 'full':
        document['training']['mode'] = 'full'
        document['training'].pop('peft')
        document['checkpointing']['output_kind'] = 'full'
    return document


def write_training_profile(path, method='lora'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(training_document(method)))
    return path


def register_models(repository):
    repository = Path(repository)
    document = json.loads((ROOT / 'plugins/registry.json').read_text())
    for name, registration in document['projects'].items():
        registration['source']['path'] = str(ROOT / 'plugins' / name)
    (repository / 'plugins').mkdir(parents=True, exist_ok=True)
    (repository / 'plugins/registry.json').write_text(json.dumps(document))


def make_repository(repository):
    repository = Path(repository)
    shutil.copytree(ROOT / 'config', repository / 'config')
    (repository / 'configs').mkdir(parents=True)
    (repository / 'experiments/training').mkdir(parents=True)
    (repository / 'experiments/evaluation').mkdir(parents=True)
    register_models(repository)
    shutil.copytree(ROOT / "examples", repository / "examples")
    return repository
