#!/usr/bin/env python3
"""Refresh the offline workflow builder from validated repository configuration."""

from __future__ import annotations

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
from ovlab_benchctl.training_composition import load_training_document
from ovlab_benchctl.policy_plugins import plugin_config_roots, plugin_project_root, registry_scope, registry_document


def build_catalog(root=ROOT):
    with registry_scope(root):
        return _build_catalog(root)


def _build_catalog(root):
    bundle = ConfigBundleBuilder(root)
    catalog = {'version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'training': [], 'evaluation': [], 'workflows': [], 'components': {}, 'tasks': {}, 'excluded': [],
               'plugins': registry_document(root)['plugins'], 'benchmarks': registry_document(root)['benchmarks']}
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

    for base in (root / 'examples/experiments', root / 'experiments'):
        for path in sorted(base.rglob('*.yaml')):
            if '.generated' in path.parts:
                continue
            reference = path.relative_to(root).as_posix()
            kind = load(path).get('kind')
            if kind == 'training':
                collect(path, lambda p=path: load_training_document(p, repository_root=root),
                        catalog['training'], reference)
            elif kind == 'experiment':
                collect(path, lambda r=reference: bundle.load_component(r, 'experiment'), catalog['evaluation'], reference)
            elif kind == 'workflow':
                def workflow(p=path):
                    document = load(p)
                    _validate_node(document)
                    return document
                collect(path, workflow, catalog['workflows'], reference)
    for kind, directory, schema in [('benchmark', 'benchmarks', 'benchmark'), ('policy', 'policies', 'policy'),
                                     ('protocol', 'protocols', 'protocol'), ('metrics', 'metrics', 'metric_set'),
                                     ('artifacts', 'artifacts', 'artifact_store')]:
        catalog['components'][kind] = []
        for base, prefix in ((root / 'config', 'ovlab://'), (root / 'configs', '')):
            for path in sorted((base / directory).rglob('*.yaml')):
                reference = prefix + path.relative_to(base).as_posix()
                collect(path, lambda r=reference, s=schema: bundle.load_component(r, s),
                        catalog['components'][kind], reference)
    component_kinds = {'benchmark': 'benchmark', 'policy': 'policy', 'protocol': 'protocol',
                       'metric_set': 'metrics', 'artifact_store': 'artifacts'}
    from ovlab_benchctl.benchmark_plugins import extension_catalog
    extensions = extension_catalog(root)
    project_roots = plugin_config_roots(root)
    for project, config_root in project_roots.items():
        files.add(extensions.project_root(project) / extensions.project_metadata(project)['manifest'])
        for path in sorted(config_root.rglob('*.yaml')):
            if path.is_symlink() or not path.resolve().is_relative_to(config_root):
                continue
            if 'examples' in path.relative_to(config_root).parts:
                continue
            files.add(path)
            try:
                kind = load(path).get('kind')
            except ValueError:
                continue
            if kind in component_kinds:
                reference = f'plugin://{project}/{path.relative_to(config_root).as_posix()}'
                def component(r=reference, s=kind):
                    document = bundle.load_component(r, s, validate_document=False)
                    if s == 'policy':
                        from ovlab_benchctl.policy_plugins import plugin_for_type, call_plugin
                        owner = plugin_for_type(document['type'], root, provider=document.get('provider'))
                        call_plugin(owner, 'validate_checkpoint_template', document, r)
                    return document
                collect(path, component,
                        catalog['components'][component_kinds[kind]], reference)
            elif kind == 'training':
                reference = f'plugin://{project}/{path.relative_to(config_root).as_posix()}'
                collect(path, lambda p=path: load_training_document(p, repository_root=root), catalog['training'], reference)
            elif kind == 'experiment':
                reference = f'plugin://{project}/{path.relative_to(config_root).as_posix()}'
                collect(path, lambda r=reference: bundle.load_component(r, 'experiment', validate_document=False),
                        catalog['evaluation'], reference)
    # Optional table metadata belongs to the registered project. Reading the
    # pinned JSON resource does not import a simulator or inspect external/.
    for project in project_roots:
        manifest = extensions.project_manifest(project)
        project_root = extensions.project_root(project)
        for provider in manifest['providers'].values():
            declaration = provider.get('authoring', {}).get('task_groups')
            if not declaration:
                continue
            path = (project_root / declaration['path']).resolve()
            if not path.is_relative_to(project_root) or not path.is_file():
                raise ValueError('authoring task table escapes or is missing from its project')
            files.add(path)
            table = json.loads(path.read_text())
            rows = table[declaration['collection']]
            for row in rows:
                group, label = row[declaration['group_by']], row[declaration['label']]
                if not isinstance(group, str) or not isinstance(label, str):
                    raise ValueError('authoring task groups and labels must be strings')
                labels = catalog['tasks'].setdefault(group, [])
                if label not in labels:
                    labels.append(label)
    # Include all component inputs (also inherited parents) and planner rules in
    # the snapshot identity, without importing any model or GPU implementation.
    files.update((root / 'config').rglob('*.yaml'))
    files.update((root / 'configs').rglob('*.yaml'))
    for name in ('schema.py', 'experiment_workflow.py', 'training_profiles.py', 'workflow_checkpoints.py'):
        files.add(root / 'code/apps/benchctl/src/ovlab_benchctl' / name)
    digest = hashlib.sha256()
    for path in sorted(files):
        if '.generated' not in path.parts:
            if path.is_relative_to(root):
                reference = path.relative_to(root).as_posix()
            elif path.name in {'plugin.json', 'extension.json'}:
                reference = 'plugin-manifest:' + load_manifest_id(path)
            else:
                owner = next((project for project in project_roots if path.is_relative_to(extensions.project_root(project))), None)
                reference = ('project-source:' + owner + '/' + path.relative_to(extensions.project_root(owner)).as_posix()) if owner else bundle.references.reference(path)
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
