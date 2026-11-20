/* Run with: node --test code/apps/workflow-builder/core.test.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const C = require('./core.js');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'catalog.js'), 'utf8'), context);
const catalog = JSON.parse(JSON.stringify(context.window.OVLAB_BUILDER_CATALOG));
const preset = name => C.clone(catalog.workflows.find(item => item.ref.endsWith('/' + name)).document);
const errors = document => C.analyze(document, catalog).issues.filter(item => item.severity === 'error');
function branches() {
  const document = preset('plugin-pipeline-smoke.yaml');
  document.execute = [];
  document.config.train.profile = 'plugin://openvla/train/default.yaml';
  document.config.evaluate = C.clone(preset('openvla-eval.yaml').config.evaluate);
  document.config.evaluate.checkpoint_from = 'lora';
  C.set(document, 'config.evaluate.component_patches.benchmark.settings.task_indices', [0, 1]);
  C.set(document, 'config.evaluate.component_patches.protocol.execution.maximum_episode_steps', 4);
  document.children = [{id: 'lora', execute: ['train'], children: [
    {id: 'full', execute: ['evaluate']},
    {id: 'eight-bit', execute: ['evaluate'], config: {evaluate: {component_patches: {policy: {settings: {runtime: {quantization: '8bit'}}}}}}},
    {id: 'four-bit', execute: ['evaluate'], config: {evaluate: {component_patches: {policy: {settings: {runtime: {quantization: '4bit'}}}}}}}
  ]}];
  return document;
}


test('every repository workflow survives YAML export and strict re-import', () => {
  for (const item of catalog.workflows) assert.deepEqual(C.parse(C.dumpYaml(item.document)), item.document, item.ref);
});

test('root checkpoint handoff schedules the source before its evaluation', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  const analysis = C.analyze(document, catalog);
  assert.deepEqual(errors(document), []);
  assert.deepEqual(analysis.rows.map(row => [row.path, row.phase, row.checkpoint]), [['', 'train', undefined], ['evaluate-checkpoint', 'evaluate', '']]);
});

test('configuration inherits and profile changes retain inherited patches', () => {
  const document = branches();
  const entry = C.flatten(document).find(item => item.path === 'lora/eight-bit');
  const resolved = C.resolve(entry, 'evaluate', catalog);
  assert.deepEqual(resolved.components.benchmark.settings.task_indices, [0, 1]);
  assert.equal(resolved.components.policy.settings.runtime.quantization, '8bit');
  assert.equal(resolved.components.protocol.execution.maximum_episode_steps, 4);
  assert.equal(entry.state.evaluate.checkpoint_from, 'lora');
  assert.equal(document.children[0].children[1].config.evaluate.checkpoint_from, undefined);
});

test('a group does not execute its inherited configurations', () => {
  const document = branches();
  assert.equal(C.analyze(document, catalog).rows.length, 4);
  document.children[0].execute = [];
  assert.ok(errors(document).some(item => item.message.includes('does not execute')));
});

test('duplicate parent and descendant evaluations have an actionable warning', () => {
  const document = branches();
  document.children[0].execute = ['train', 'evaluate'];
  assert.ok(C.analyze(document, catalog).issues.some(item => item.severity === 'warning' && item.path === 'lora' && item.message.includes('lora/full')));
});

test('source method and QuIC quantization are checked before export', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  C.set(document.children[0], 'config.evaluate.component_patches.policy.settings.runtime.quantization', '4bit');
  assert.ok(errors(document).some(item => item.message.includes('quantization')));
  delete document.children[0].config;
  document.config.evaluate = C.clone(preset('openvla-eval.yaml').config.evaluate);
  document.config.evaluate.checkpoint_from = '';
  assert.ok(errors(document).some(item => item.message.includes('incompatible')));
});

test('dependency cycles and missing phases are rejected', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  document.depends_on = [{node: 'evaluate-checkpoint', phase: 'evaluate'}];
  assert.ok(errors(document).some(item => item.message.includes('cycle')));
  delete document.depends_on;
  document.config.evaluate.checkpoint_from = 'missing';
  assert.ok(errors(document).some(item => item.message.includes('does not execute')));
});

test('renaming and moving a subtree preserve checkpoint/dependency references', () => {
  const document = branches();
  document.children.push({id: 'models', name: 'Models'});
  const next = C.moveNode(document, 'lora', 'models');
  assert.equal(next, 'models/lora');
  const child = C.flatten(document).find(item => item.path === 'models/lora/full');
  assert.equal(child.state.evaluate.checkpoint_from, 'models/lora');
  C.renameNode(document, 'models', 'adaptations');
  assert.equal(C.flatten(document).find(item => item.path === 'adaptations/lora/full').state.evaluate.checkpoint_from, 'adaptations/lora');
  assert.deepEqual(errors(document), []);
  assert.throws(() => C.moveNode(document, 'adaptations', 'adaptations/lora'), /descendants/);
});

test('reserved IDs, sibling collisions and invalid task indices are rejected', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  assert.throws(() => C.renameNode(document, 'evaluate-checkpoint', 'evaluate'), /reserved/);
  document.children.push({id: 'other'});
  assert.throws(() => C.renameNode(document, 'evaluate-checkpoint', 'other'), /sibling/);
  C.set(document.children[0], 'config.evaluate.component_patches.benchmark.settings.task_indices', [10]);
  assert.ok(errors(document).some(item => item.message.includes('Task indices')));
});

test('unsafe JSON/YAML keys and duplicate YAML mappings are rejected', () => {
  assert.throws(() => C.parse('{"__proto__":{"polluted":true}}'), /Unsupported key/);
  assert.throws(() => C.parse('{"id":"one","id":"two"}'), /Duplicate JSON key/);
  assert.throws(() => C.parse('{"nested":{"id":"one","id":"two"}}'), /Duplicate JSON key/);
  assert.throws(() => C.parse('id: one\nid: two\n'), /Duplicate key/);
  assert.throws(() => C.parse('id: &anchor value\n'), /anchors/);
  assert.throws(() => C.parse('id:\n   bad: indentation\n'), /indentation/);
  assert.equal({}.polluted, undefined);
});

test('scalar syntax preserves quotes, comments, booleans and small numbers', () => {
  const document = {id: 'a # name', name: 'a: b', enabled: true, rate: 0.000002, empty: {}, choices: ['true', '0', 'a,b']};
  assert.deepEqual(C.parse(C.dumpYaml(document)), document);
  assert.deepEqual(C.parse('id: one # comment\nname: "a # b"\n'), {id: 'one', name: 'a # b'});
});

test('real evaluation catalog includes working native mock configurations', () => {
  for (const source of catalog.evaluation.filter(item => item.document.components.benchmark)) {
    const document = {schema_version: 'ovlab.experiment/v1', kind: 'workflow', id: 'catalog-check', name: 'Catalog', execute: ['evaluate'], config: {evaluate: {profile: source.ref}}};
    assert.deepEqual(errors(document), [], source.ref);
  }
});

test('invalid advanced settings return errors without crashing the editor', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  C.set(document, 'config.train.patch.training', null);
  assert.ok(errors(document).some(item => item.message.includes('preserve')));
  assert.equal(C.analyze({...document, children: Array.from({length: 513}, (_, index) => ({id: 'n' + index}))}, catalog).entries.length, 0);
});

test('blocked dependency cycles retain the complete phase inventory', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  document.depends_on = [{node: 'evaluate-checkpoint', phase: 'evaluate'}];
  const analysis = C.analyze(document, catalog);
  assert.equal(analysis.phaseCount, 2);
  assert.equal(analysis.rows.length, 0);
  assert.equal(analysis.blockedRows.length, 2);
  assert.ok(analysis.blockedRows.every(row => row.blocked));
});

test('different labels do not hide duplicate evaluation work', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  document.execute.push('evaluate');
  C.set(document.children[0], 'config.evaluate.patch.experiment', {id: 'renamed-evaluation', name: 'A new label', tags: ['different-label']});
  const warning = C.analyze(document, catalog).issues.find(issue => issue.fix === 'remove-evaluate');
  assert.equal(warning.path, '');
  assert.equal(warning.related, 'evaluate-checkpoint');
});

test('malformed policy types and training methods produce actionable errors', () => {
  for (const type of [7, null, {}, 'unsupported']) {
    const document = preset('plugin-pipeline-smoke.yaml');
    C.set(document, 'config.evaluate.component_patches.policy.type', type);
    assert.ok(errors(document).some(issue => issue.message.includes('policy type')));
  }
  const document = preset('plugin-pipeline-smoke.yaml');
  C.set(document, 'config.train.patch.training.peft.method', 7);
  assert.ok(errors(document).some(issue => issue.message.includes('supported method')));
});

test('numeric IDs are rejected and long duplicate IDs remain portable', () => {
  const document = preset('plugin-pipeline-smoke.yaml');
  document.id = 123;
  assert.equal(C.analyze(document, catalog).entries.length, 0);
  const prefix = 'a'.repeat(120) + '-copy';
  const first = C.uniqueId({children: []}, prefix);
  const second = C.uniqueId({children: [{id: first}]}, prefix);
  assert.notEqual(first, second);
  assert.ok(C.SAFE.test(first) && C.SAFE.test(second));
});


test('new plugin methods, policy types and quantization come from descriptors', () => {
  const next = C.clone(catalog), document = preset('plugin-pipeline-smoke.yaml');
  const policyRef = 'plugin://future-model/policy/default.yaml';
  const trainRef = 'plugin://future-model/train/default.yaml';
  const policy = C.clone(catalog.components.policy.find(item => item.ref === 'plugin://openvla-quic/policy/default.yaml').document);
  policy.type = 'future_policy';
  next.components.policy.push({ref: policyRef, document: policy});
  const training = C.clone(catalog.training.find(item => item.ref === 'plugin://openvla-quic/train/default.yaml').document);
  training.trainer = {id: 'future-model', implementation: 'future-trainer'};
  training.training.peft.method = 'future-method';
  next.training.push({ref: trainRef, document: training});
  next.plugins.future_policy = {id: 'future', project: 'future-model', family: 'future-family',
    deployment: {profile: 'future-model', quantization: ['none', 'int2']}, capabilities: {
      training: {implementations: ['future-trainer'], methods: ['future-method']},
      checkpoint_binding: {training_methods: ['future-method']}}};
  document.config.train.profile = trainRef;
  document.config.evaluate.patch.components.policy = policyRef;
  document.config.evaluate.patch.deployment = {profile: 'future-model', renderer: 'egl'};
  C.set(document, 'config.evaluate.component_patches.policy.settings.runtime.quantization', 'int2');
  assert.deepEqual(C.analyze(document, next).issues.filter(issue => issue.severity === 'error'), []);
  C.set(document, 'config.evaluate.component_patches.policy.settings.runtime.quantization', '4bit');
  assert.ok(C.analyze(document, next).issues.some(issue => issue.message.includes('quantization')));
});

test('partial plugin defaults require explicit scenario composition', () => {
  const document = {schema_version: 'ovlab.experiment/v1', kind: 'workflow', id: 'partial', name: 'Partial',
    execute: ['evaluate'], config: {evaluate: {profile: 'plugin://openvla/evaluate/default.yaml'}}};
  assert.ok(errors(document).some(issue => issue.message.includes('experiment details')));
});
