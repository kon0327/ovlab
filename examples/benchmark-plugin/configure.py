#!/usr/bin/env python3
"""Prepare a private CPU evaluation configuration without changing root registration."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def configure(root, *, data_root=None, dry_run=False):
    root = Path(root).resolve()
    package = root / 'examples/benchmark-plugin'
    checkpoint = root / 'examples/policy-plugin/fixtures/checkpoint'
    if not checkpoint.is_dir():
        raise ValueError(f'example checkpoint is missing: {checkpoint}')
    data = Path(data_root or Path(tempfile.gettempdir()) / 'ovlab-cpu-plugin-demo').expanduser().resolve()
    registry = json.loads((package / 'registry.json').read_text())
    profile = f'''schema_version: "0.1.0"
kind: local_profile
id: cpu-plugin-example-local
paths:
  checkpoint_root: {json.dumps(str(data / 'checkpoints'))}
  dataset_root: {json.dumps(str(data / 'datasets'))}
  runs_root: {json.dumps(str(data / 'runs'))}
devices:
  primary_gpu: cpu
resources:
  checkpoints:
    example-constant:
      local_path: {json.dumps(str(checkpoint))}
'''
    compose_values = {
        'OVLAB_RUNS_ROOT': str(data / 'runs'),
        'OVLAB_EXPORTS_ROOT': str(data / 'exports'),
        'OVLAB_CONFIGS_ROOT': str(root / 'configs'),
        'OVLAB_HOST_ARTIFACT_GID': str(os.getgid()),
        'OVLAB_REPORTING_IMAGE': 'ovlab-reporting:local',
    }
    env = ''.join(f'{key}={json.dumps(value.replace("$", "$$"))}\n' for key, value in compose_values.items())
    files = {
        root / 'configs/extensions/cpu-example.json': json.dumps(registry, indent=2) + '\n',
        root / 'configs/evaluation/cpu-plugin-example.yaml': 'schema_version: "0.1.0"\nkind: experiment\nextends: plugin://example-benchmark/evaluate.yaml\n',
        root / 'configs/local/cpu-plugin-example.yaml': profile,
        root / 'configs/local/cpu-plugin-example.env': env,
    }
    for target, content in files.items():
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'example configuration escapes the repository: {target}')
        if target.exists() and (not target.is_file() or target.read_text() != content):
            raise ValueError(f'preserving modified configuration: {target}')
    if not dry_run:
        for target, content in files.items():
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('x') as stream:
                    stream.write(content)
    return [str(path.relative_to(root)) for path in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for path in configure(args.root, data_root=args.data_root, dry_run=args.dry_run):
        print(path)


if __name__ == '__main__':
    main()
