#!/usr/bin/env python3
"""Refresh the offline workflow builder from validated repository configuration."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'code/apps', ROOT / 'code/packages'):
    for source in sorted(directory.glob('*/src')):
        sys.path.insert(0, str(source))

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.experiment_workflow import _validate_node
from ovlab_benchctl.strict_yaml import load
from ovlab_benchctl.training_composition import load_training_profile
from ovlab_benchctl.policy_plugins import plugin_config_roots, plugin_project_root, registry_scope


def build_catalog(root=ROOT):
    with registry_scope(root):
        return _build_catalog(root)


def _build_catalog(root):
    bundle = ConfigBundleBuilder(root)
    catalog = {'version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'training': [], 'evaluation': [], 'workflows': [], 'components': {}, 'tasks': {}, 'excluded': []}
    files = set()

    def collect(path, callback, destination, reference):
        if '.generated' in path.parts or path.name == 'example.yaml':
            return
        files.add(path)
        try:
            document = callback()
            destination.append({'ref': reference, 'label': document.get('id') or
                                document.get('experiment', {}).get('name') or path.stem,
                                'document': document})
        except (ValueError, RuntimeError, OSError) as exc:
            catalog['excluded'].append({'ref': reference, 'reason': str(exc)})

    for path in sorted((root / 'experiments/training').rglob('*.yaml')):
        collect(path, lambda p=path: dict(load_training_profile(p).document), catalog['training'],
                path.relative_to(root).as_posix())
    for path in sorted((root / 'experiments/evaluation').rglob('*.yaml')):
        reference = path.relative_to(root).as_posix()
        collect(path, lambda r=reference: bundle.load_component(r, 'experiment'), catalog['evaluation'], reference)
    for kind, directory, schema in [('benchmark', 'benchmarks', 'benchmark'), ('policy', 'policies', 'policy'),
                                     ('protocol', 'protocols', 'protocol'), ('metrics', 'metrics', 'metric_set'),
                                     ('artifacts', 'artifacts', 'artifact_store')]:
        catalog['components'][kind] = []
        for path in sorted((root / 'configs' / directory).rglob('*.yaml')):
            reference = path.relative_to(root / 'configs').as_posix()
            collect(path, lambda r=reference, s=schema: bundle.load_component(r, s),
                    catalog['components'][kind], reference)
    component_kinds = {'benchmark': 'benchmark', 'policy': 'policy', 'protocol': 'protocol',
                       'metric_set': 'metrics', 'artifact_store': 'artifacts'}
    project_roots = plugin_config_roots(root)
    for project, config_root in project_roots.items():
        files.add(plugin_project_root(project, root) / 'plugin.json')
        for path in sorted(config_root.rglob('*.yaml')):
            if path.is_symlink() or not path.resolve().is_relative_to(config_root):
                continue
            files.add(path)
            try:
                kind = load(path).get('kind')
            except ValueError:
                continue
            if kind in component_kinds:
                reference = f'plugin://{project}/{path.relative_to(config_root).as_posix()}'
                collect(path, lambda r=reference, s=kind: bundle.load_component(r, s),
                        catalog['components'][component_kinds[kind]], reference)
    for path in sorted((root / 'experiments/workflows').glob('*')):
        if path.suffix not in {'.yaml', '.yml', '.json'}:
            continue
        def workflow(p=path):
            document = json.loads(p.read_text()) if p.suffix == '.json' else load(p)
            _validate_node(document)
            return document
        collect(path, workflow, catalog['workflows'], path.relative_to(root).as_posix())
    task_source = root / 'external/libero/libero/libero/benchmark/libero_suite_task_map.py'
    if task_source.is_file():
        files.add(task_source)
        for statement in ast.parse(task_source.read_text()).body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in statement.targets):
                catalog['tasks'] = ast.literal_eval(statement.value)
                break
    # Include all component inputs (also inherited parents) and planner rules in
    # the snapshot identity, without importing any model or GPU implementation.
    files.update((root / 'configs').rglob('*.yaml'))
    for name in ('schema.py', 'experiment_workflow.py', 'training_profiles.py', 'workflow_checkpoints.py'):
        files.add(root / 'code/apps/benchctl/src/ovlab_benchctl' / name)
    digest = hashlib.sha256()
    for path in sorted(files):
        if '.generated' not in path.parts:
            if path.is_relative_to(root):
                reference = path.relative_to(root).as_posix()
            elif path.name == 'plugin.json':
                reference = 'plugin-manifest:' + load_manifest_id(path)
            else:
                reference = bundle.references.reference(path)
            digest.update(reference.encode() + b'\0' + path.read_bytes())
    catalog['source_sha256'] = digest.hexdigest()
    return catalog


def load_manifest_id(path):
    return json.loads(path.read_text())['id']


if __name__ == '__main__':
    catalog = build_catalog()
    target = ROOT / 'code/apps/workflow-builder/catalog.js'
    target.parent.mkdir(parents=True, exist_ok=True)
    # Script, rather than fetch(), lets the complete editor work through file://.
    target.write_text('// Generated by deploy/scripts/build-workflow-catalog.py; do not edit.\n'
                      + 'window.OVLAB_BUILDER_CATALOG = ' + json.dumps(catalog, ensure_ascii=False, separators=(',', ':'))
                      .replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029') + ';\n')
    print(f'{target.relative_to(ROOT)}: {len(catalog["training"])} training profiles, '
          f'{len(catalog["evaluation"])} evaluation profiles, {len(catalog["workflows"])} workflows; '
          f'{len(catalog["excluded"])} unsupported examples excluded')
