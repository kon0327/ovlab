"""Isolated, offline preparation of dense inference weights from training bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


CONVERTER_VERSION = 'ovlab-dense-inference/v1'
IMAGE_CAPABILITY_LABEL = 'io.github.ovlab.checkpoint-preparation'
IMAGE_CAPABILITY = 'lora-merge-v1'


def ensure_weight_index(snapshot: Path) -> None:
    """HF only emits an index for sharded models; our artifact contract always uses one."""
    from ovlab_policy_sdk.training_support import safetensors_header as _safetensors_header
    index = snapshot / 'model.safetensors.index.json'
    if index.is_file():
        return
    weight_map, total = {}, 0
    for path in sorted(snapshot.glob('*.safetensors')):
        header, _ = _safetensors_header(path)
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if name in weight_map:
                raise ValueError(f'duplicate dense tensor: {name}')
            weight_map[name] = path.name
            total += tensor['data_offsets'][1] - tensor['data_offsets'][0]
    if not weight_map:
        raise ValueError('dense inference snapshot has no Safetensors weights')
    index.write_text(json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, sort_keys=True))


def finish_snapshot(snapshot: Path, bundle: Path) -> None:
    """Retain the trained normalization and processor, never modify the base model."""
    for path in sorted((bundle / 'processor').glob('*')):
        if path.is_symlink() or not path.is_file():
            raise ValueError('processor entries must be regular files')
        shutil.copyfile(path, snapshot / path.name)
    statistics = bundle / 'auxiliary/dataset_statistics.json'
    if not statistics.is_file() or statistics.is_symlink():
        raise ValueError('training checkpoint has no frozen dataset statistics')
    stats = json.loads(statistics.read_text())
    if not isinstance(stats, dict) or not stats:
        raise ValueError('training checkpoint has invalid dataset statistics')
    shutil.copyfile(statistics, snapshot / 'dataset_statistics.json')
    config_path = snapshot / 'config.json'
    config = json.loads(config_path.read_text())
    config['norm_stats'] = {**config.get('norm_stats', {}), **stats}
    config_path.write_text(json.dumps(config, sort_keys=True))
    if (snapshot / 'adapter_config.json').exists():
        raise ValueError('dense inference snapshot must not contain an active PEFT configuration')
    ensure_weight_index(snapshot)


def merge_lora(base: Path, bundle: Path, output: Path, *, dtype='bfloat16') -> None:
    """Merge on CPU before inference quantization, using the serving model implementation."""
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise ValueError('checkpoint preparation requires offline mode')
    weights = bundle / 'weights-or-adapter'
    if not (weights / 'adapter_model.safetensors').is_file() or any(weights.glob('*.bin')):
        raise ValueError('LoRA preparation requires the canonical Safetensors adapter')
    from ovlab_openvla_vanilla.runtime import HuggingFaceOpenVlaRuntime
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    torch, _, _, auto_config, auto_model, auto_processor, _ = HuggingFaceOpenVlaRuntime._runtime_imports()
    config = auto_config.from_pretrained(base, trust_remote_code=False, local_files_only=True)
    model = auto_model.from_pretrained(
        base, config=config, torch_dtype={'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype], low_cpu_mem_usage=True,
        trust_remote_code=False, local_files_only=True, attn_implementation='eager',
    )
    model = PeftModel.from_pretrained(model, weights, is_trainable=False)
    expected = load_file(weights / 'adapter_model.safetensors', device='cpu')
    loaded = get_peft_model_state_dict(model)
    if set(expected) != set(loaded) or any(
            tensor.shape != loaded[name].shape or
            not torch.equal(loaded[name].detach().cpu(), tensor.to(loaded[name].dtype))
            for name, tensor in expected.items()):
        raise ValueError('LoRA checkpoint tensors were not completely and exactly loaded')
    del expected, loaded
    model = model.merge_and_unload(safe_merge=True)
    if any('lora_' in name.lower() for name, _ in model.named_parameters()):
        raise ValueError('LoRA merge left active adapter parameters')
    model.eval()
    model.save_pretrained(output, safe_serialization=True, max_shard_size='2GB')
    processor_path = bundle / 'processor'
    processor = auto_processor.from_pretrained(
        processor_path if (processor_path / 'preprocessor_config.json').is_file() else base,
        config=config, trust_remote_code=False, local_files_only=True,
    )
    processor.save_pretrained(output)
    finish_snapshot(output, bundle)
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o444)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    args = parser.parse_args(argv)
    os.umask(0o007)
    merge_lora(args.base, args.checkpoint, args.output, dtype=args.dtype)
    print(json.dumps({'status': 'completed', 'converter': CONVERTER_VERSION}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
