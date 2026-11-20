#!/usr/bin/env python3
"""Prepare the example's namespaced configs; never overwrite developer changes."""
import argparse
import json
from pathlib import Path


def configure(root, *, dry_run=False):
    root = Path(root).resolve()
    package = root / 'examples/policy-plugin'
    files = {
        root / 'configs/policies/example-plugin.yaml': 'schema_version: \"0.1.0\"\nkind: policy\nextends: plugin://example-policy/policy.yaml\n',
        root / 'configs/deploy/example-plugin.yaml': 'schema_version: ovlab.deployment/v1\nkind: deployment\nextends: plugin://example-policy/deployment.yaml\ncomponents:\n  policy: policies/example-plugin.yaml\nresources:\n  registry: resources/example-plugin.yaml\n',
        root / 'configs/resources/example-plugin.yaml': 'schema_version: \"0.1.0\"\nkind: resource_registry\nextends: plugin://example-policy/resources.yaml\n',
    }
    checkpoint = package / 'fixtures/checkpoint'
    if not checkpoint.is_dir():
        raise ValueError(f'example checkpoint is missing: {checkpoint}')
    data = root.parent / 'ovlab-data'
    profile = f'''schema_version: "0.1.0"
kind: local_profile
id: example-plugin-local
paths:
  checkpoint_root: {json.dumps(str(data / 'checkpoints/huggingface'))}
  dataset_root: {json.dumps(str(data / 'datasets/libero'))}
  runs_root: {json.dumps(str(data / 'runs'))}
devices:
  primary_gpu: cpu
resources:
  checkpoints:
    example-constant:
      local_path: {json.dumps(str(checkpoint))}
'''
    files[root / 'configs/local/example-plugin.yaml'] = profile
    for target, content in files.items():
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'example configuration escapes the repository: {target}')
        if target.exists() and target.read_text() != content:
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
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for path in configure(args.root, dry_run=args.dry_run):
        print(path)


if __name__ == '__main__':
    main()
