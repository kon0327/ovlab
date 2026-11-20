(function () {
  'use strict';
  const C = window.OVLABBuilder, catalog = window.OVLAB_BUILDER_CATALOG;
  const $ = selector => document.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const storageKey = 'ovlab.workflow-studio.v1';
  if (!catalog || !C || !window.OVLABStudioDOM) { document.body.innerHTML = '<p class="fatal-error">Builder files are missing. Keep all editor files together and run python deploy/scripts/build-workflow-catalog.py from the repository.</p>'; return; }
  const own = C.own;
  const methodLabel = method => String(method || '').replace(/_/g, ' ');
  let workflow, selected = '', view = 'builder', zoom = 1, analysis, sourceDirty = false, toastTimer;
  let advancedDraft = null, advancedError = '', restored = false, storageAvailable = true;
  const history = [], future = [];
  const blank = () => ({schema_version: 'ovlab.experiment/v1', kind: 'workflow', id: 'new-experiment', name: 'New experiment', execution: {max_workers: 1}, children: []});
  const defaultPreset = catalog.workflows.find(item => C.flatten(item.document).some(node => (node.node.execute || []).includes('train'))) || catalog.workflows[0];
  const phaseConfiguration = state => Object.fromEntries(['profile', 'patch', 'component_patches', 'checkpoint_from'].filter(key => own(state, key)).map(key => [key, C.clone(state[key])]));
  function referenceConfigurations(phase) {
    const entries = catalog.workflows.flatMap(item => C.flatten(item.document));
    for (const source of catalog[phase === 'train' ? 'training' : 'evaluation']) entries.push({state: {[phase]: {profile: source.ref}}});
    return entries.map(entry => ({resolved: C.resolve(entry, phase, catalog), config: phaseConfiguration(entry.state[phase] || {})})).filter(item => item.resolved && !item.resolved.unavailable);
  }
  function evaluationFor(trained) {
    const choices = referenceConfigurations('evaluate');
    return choices.find(item => C.bindingCompatible(trained, item.resolved)) || choices.find(item => item.resolved.components.policy?.type === 'mock') || choices[0];
  }
  workflow = C.clone(defaultPreset?.document || blank());
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (saved?.version === 1 && C.analyze(C.guard(saved.document), catalog).entries.length) {
      workflow = saved.document; selected = saved.selected || ''; restored = true;
      if (typeof saved.source === 'string' && saved.source.length <= 2000000) { $('#yaml-source').value = saved.source; sourceDirty = true; }
      else if (typeof saved.advanced?.text === 'string' && saved.advanced.text.length <= 2000000 && C.flatten(workflow).some(item => item.path === saved.advanced.path)) advancedDraft = saved.advanced;
    }
  } catch (_) { /* Storage is optional in file:// and private browsing. */ }

  function toast(message) { clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false; toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4200); }
  const hasDraft = () => sourceDirty || advancedDraft !== null;
  function snapshot() { return {document: C.clone(workflow), selected, source: sourceDirty ? $('#yaml-source').value : null, advanced: advancedDraft ? C.clone(advancedDraft) : null}; }
  function restore(state) {
    workflow = state.document; selected = state.selected; sourceDirty = typeof state.source === 'string';
    if (sourceDirty) $('#yaml-source').value = state.source;
    advancedDraft = state.advanced || null; advancedError = '';
  }
  function persist() {
    let message;
    try { localStorage.setItem(storageKey, JSON.stringify({version: 1, ...snapshot()})); storageAvailable = true; message = hasDraft() ? 'Draft saved · not applied' : 'Saved in this browser'; }
    catch (_) { storageAvailable = false; message = 'Storage unavailable · export to save'; }
    if ($('#save-state').textContent !== message) $('#save-state').textContent = message;
  }
  function canChange(allowDraft) {
    if (allowDraft === 'replace' || (!sourceDirty && !advancedDraft) || (allowDraft === 'source' && !advancedDraft) || (allowDraft === 'advanced' && !sourceDirty)) return true;
    toast('Apply or discard the saved editor draft before changing the workflow.');
    return false;
  }
  function change(action, allowDraft) {
    if (!canChange(allowDraft)) return false;
    const before = snapshot();
    try {
      action();
      const result = C.analyze(workflow, catalog);
      if (!result.entries.length) throw new Error(result.issues.find(issue => issue.severity === 'error')?.message || 'Invalid workflow structure.');
      if (['source', 'advanced', 'replace'].includes(allowDraft)) { sourceDirty = false; advancedDraft = null; advancedError = ''; }
      if (!C.sameDocument(before.document, workflow)) {
        if (allowDraft !== 'replace') { before.source = null; before.advanced = null; }
        history.push(before); if (history.length > 80) history.shift(); future.length = 0;
      }
      render(); persist(); return true;
    } catch (error) { restore(before); toast(error.message); render(); return false; }
  }
  function undo(redo = false) {
    if (!canChange()) return;
    const from = redo ? future : history, to = redo ? history : future;
    if (!from.length) return;
    to.push(snapshot()); restore(from.pop()); render(); persist();
  }
  function entry() { return C.flatten(workflow).find(item => item.path === selected) || C.flatten(workflow)[0]; }
  function confirmAction(title, description, actionLabel = 'Continue') {
    const dialog = $('#confirm-dialog'); $('#confirm-title').textContent = title; $('#confirm-description').textContent = description; $('#confirm-action').textContent = actionLabel;
    dialog.returnValue = 'cancel';
    return new Promise(resolve => { dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), {once: true}); dialog.showModal(); });
  }
  async function replaceWorkflow(document, message) {
    const result = C.analyze(document, catalog);
    if (!result.entries.length) throw new Error(result.issues.find(item => item.severity === 'error')?.message || 'Import an OVLAB workflow YAML or JSON document.');
    if ((history.length || hasDraft() || restored) && !await confirmAction('Replace this workflow?', 'Your current workflow and any unapplied draft can be restored with Undo. Export first to keep a separate file.', 'Replace workflow')) return;
    change(() => { workflow = C.clone(document); selected = ''; }, 'replace'); restored = false;
    view = 'builder'; render(); requestAnimationFrame(() => fitCanvas(false)); toast(message);
  }
  const phasePills = node => (node.execute || []).map(phase => `<span class="pill ${phase}">${phase === 'train' ? 'Train' : 'Evaluate'}</span>`).join('') || '<span class="pill group">Group only</span>';

  function renderOutline() {
    window.OVLABStudioDOM.render($('#outline'), analysis.entries.map(item => `<button class="outline-row${item.path === selected ? ' selected' : ''}" role="treeitem" tabindex="${item.path === selected ? 0 : -1}" aria-level="${item.depth + 1}" aria-selected="${item.path === selected}" ${item.node.children?.length ? 'aria-expanded="true"' : ''} title="${esc(item.path || item.node.id)}" data-select="${esc(item.path)}" data-drop="${esc(item.path)}" ${item.path ? 'draggable="true"' : ''} style="padding-left:${8 + item.depth * 13}px"><span class="mini-icon" aria-hidden="true">${item.node.children?.length ? '▦' : '◇'}</span><span class="outline-label">${esc(item.node.id)}</span><span class="outline-phase">${(item.node.execute || []).map(phase => `<span class="${phase === 'train' ? 'train-icon' : 'eval-icon'}">${phase === 'train' ? 'T' : 'E'}</span>`).join('')}</span></button>`).join(''));
    $('#node-count').textContent = analysis.entries.length;
  }
  function renderCanvas() {
    const points = new Map(); let leaf = 0;
    function layout(node, path, depth) {
      const children = node.children || [], positions = children.map(child => layout(child, path ? `${path}/${child.id}` : child.id, depth + 1));
      const y = positions.length ? (positions[0] + positions.at(-1)) / 2 : leaf++ * 178 + 30;
      points.set(path, {x: depth * 306 + 30, y}); return y;
    }
    layout(workflow, '', 0);
    const width = Math.max(...[...points.values()].map(point => point.x)) + 282;
    const height = Math.max(1, leaf) * 178 + 40;
    let connections = '';
    for (const item of analysis.entries) if (item.parent !== null) {
      const p = points.get(item.parent), q = points.get(item.path), x = p.x + 252, y = p.y + 70;
      connections += `<path d="M${x},${y} C${x + 28},${y} ${q.x - 28},${q.y + 70} ${q.x},${q.y + 70}"/>`;
    }
    const cards = analysis.entries.map(item => {
      const point = points.get(item.path), phases = item.node.execute || [], phase = phases.includes('train') ? 'train' : phases.includes('evaluate') ? 'evaluate' : '';
      const resolved = phase && C.resolve(item, phase, catalog);
      let subtitle = phase ? `${methodLabel(C.family(resolved, phase)) || 'Choose a profile'}` : 'Settings inherited by child nodes';
      if (resolved && !resolved.unavailable) {
        const precision = resolved.components.policy?.settings.runtime?.quantization || 'native';
        subtitle += phase === 'train' ? ` · ${resolved.document.training.max_steps} optimizer steps` : ` · ${resolved.components.protocol?.execution.rollouts_per_task} episode(s) · ${precision === 'none' ? 'full' : precision}`;
      }
      const bound = phases.includes('evaluate') && own(item.state.evaluate, 'checkpoint_from');
      const count = item.node.children?.length;
      return `<button class="node-card${item.path === selected ? ' selected' : ''}${analysis.issues.some(issue => issue.path === item.path && issue.severity === 'error') ? ' error' : ''}" data-select="${esc(item.path)}" style="left:${point.x}px;top:${point.y}px" aria-label="Select ${esc(item.node.name || item.node.id)}" title="${esc((item.path || '<root>') + '\n' + (item.node.name || item.node.id))}"><span class="node-top"><span class="node-icon ${phase === 'train' ? 'train-icon' : phase === 'evaluate' ? 'eval-icon' : ''}">${phase === 'train' ? 'T' : phase === 'evaluate' ? 'E' : '▦'}</span><span class="node-title">${esc(item.node.name || item.node.id)}</span></span><span class="node-subtitle">${esc(subtitle)}</span><span class="node-bottom">${phasePills(item.node)}<span class="node-count-label">${count ? count + (count === 1 ? ' child' : ' children') : 'leaf'}</span></span>${bound ? `<span class="node-binding" title="Uses the finalized training checkpoint">↳ ${esc((item.state.evaluate.checkpoint_from || '<root>') + ':train')}</span>` : ''}</button>`;
    }).join('');
    $('#canvas').style.width = Math.max(width * zoom, $('#canvas-scroll').clientWidth - 40) + 'px';
    $('#canvas').style.height = Math.max(height * zoom, $('#canvas-scroll').clientHeight - 40) + 'px';
    window.OVLABStudioDOM.render($('#canvas'), `<div class="graph" style="position:relative;width:${width}px;height:${height}px;transform:scale(${zoom});transform-origin:top left"><svg class="connectors" aria-hidden="true" viewBox="0 0 ${width} ${height}">${connections}</svg>${cards}</div>`);
    $('#zoom-label').textContent = Math.round(zoom * 100) + '%';
  }
  function options(items, selectedValue, label = item => item.label || item.ref) {
    return items.map(item => `<option value="${esc(item.ref)}" ${item.ref === selectedValue ? 'selected' : ''}>${esc(label(item))}</option>`).join('');
  }
  function field(label, path, effective, {type = 'text', min, step, hint = '', integer = false, reset = true} = {}) {
    const local = C.get(entry().node, path), value = local === undefined ? '' : Array.isArray(local) ? local.join(', ') : local;
    const shown = Array.isArray(effective) ? effective.join(', ') : effective;
    const configurable = path.startsWith('config.') || path.startsWith('execution.');
    return `<div class="field"><div class="field-heading"><label for="field-${esc(path)}">${esc(label)}</label>${local !== undefined && reset ? `<button type="button" class="reset-field" data-reset="${esc(path)}" title="Remove this local override">Reset</button>` : ''}</div><input id="field-${esc(path)}" data-field="${esc(path)}" data-value-type="${integer ? 'integer' : type}" type="${type === 'number' ? 'number' : 'text'}" ${min !== undefined ? `min="${min}"` : ''} ${step !== undefined ? `step="${step}"` : ''} value="${esc(value)}" placeholder="${esc(shown !== undefined ? shown : 'Inherit')}" aria-describedby="help-${esc(path)}"><small id="help-${esc(path)}">${esc(hint || (configurable ? local === undefined ? 'Using the profile or parent value. Enter a value to override.' : 'Local override. Reset to use the profile or parent value.' : ''))}</small></div>`;
  }
  function profileMethod(source, phase) {
    if (phase === 'train') return source.document.training.mode === 'peft' ? source.document.training.peft.method : 'full';
    const policy = catalog.components.policy.find(item => item.ref === source.document.components.policy);
    return catalog.plugins?.[policy?.document.type]?.family || policy?.document.type || 'other';
  }
  function profileField(item, phase) {
    const local = item.node.config?.[phase]?.profile, inherited = item.parent !== null ? C.flatten(workflow).find(node => node.path === item.parent).state[phase]?.profile : undefined;
    const source = C.profile(catalog, phase, local), resolved = C.resolve(item, phase, catalog);
    const items = catalog[phase === 'train' ? 'training' : 'evaluation'];
    const groups = [...new Set(items.map(option => profileMethod(option, phase)))];
    const choices = groups.map(method => `<optgroup label="${esc(methodLabel(method))}">${options(items.filter(option => profileMethod(option, phase) === method), source?.ref || local, option => option.ref.split('/').slice(2).join('/'))}</optgroup>`).join('');
    return `<div class="field"><label for="${phase}-profile">${phase === 'train' ? 'Training' : 'Evaluation'} profile</label><select id="${phase}-profile" data-profile="${phase}"><option value="__inherit__" ${!local ? 'selected' : ''}>${esc(inherited ? 'Inherit · ' + inherited.split('/').at(-1) : 'Not configured')}</option>${local && !source ? `<option selected value="${esc(local)}">Unavailable: ${esc(local)}</option>` : ''}${choices}</select><small class="effective">${resolved && !resolved.unavailable ? esc('Effective: ' + resolved.source.ref + (item.state[phase].profileSource !== item.path ? ' · inherited from ' + (item.state[phase].profileSource || '<root>') : '')) : 'Choose a repository profile or inherit one from a parent.'}</small></div>`;
  }
  function selectField(label, path, effective, choices) {
    const local = C.get(entry().node, path);
    const unsupported = local !== undefined && !choices.some(([value]) => String(local) === String(value));
    return `<div class="field"><label for="field-${esc(path)}">${esc(label)}</label><select id="field-${esc(path)}" data-select-field="${esc(path)}"><option value="__inherit__" ${local === undefined ? 'selected' : ''}>${esc('Inherit · ' + (effective ?? 'not set'))}</option>${unsupported ? `<option value="${esc(local)}" selected disabled>Unsupported: ${esc(local)}</option>` : ''}${choices.map(([value, title]) => `<option value="${esc(value)}" ${String(local) === String(value) ? 'selected' : ''}>${esc(title)}</option>`).join('')}</select></div>`;
  }
  function renderTraining(item) {
    const resolved = C.resolve(item, 'train', catalog);
    let body = profileField(item, 'train');
    if (resolved && !resolved.unavailable) {
      const doc = resolved.document, train = doc.training, method = C.family(resolved, 'train'), suite = doc.dataset.ref.split('/').at(-1), tasks = catalog.tasks[suite] || [];
      body += `<div class="profile-info"><strong>${esc(methodLabel(method))}</strong> · ${esc(doc.model.base_checkpoint)}<br>${esc(doc.dataset.ref)} · ${esc(train.precision)} · ${esc(train.quantization)} quantization</div>`;
      body += '<div class="field-row">' + field('Optimizer steps', 'config.train.patch.training.max_steps', train.max_steps, {type: 'number', min: 1, step: 1, integer: true}) + field('Batch size', 'config.train.patch.training.per_device_batch_size', train.per_device_batch_size, {type: 'number', min: 1, step: 1, integer: true}) + '</div>';
      body += '<div class="field-row">' + field('Gradient accumulation', 'config.train.patch.training.gradient_accumulation_steps', train.gradient_accumulation_steps, {type: 'number', min: 1, step: 1, integer: true}) + field('Learning rate', 'config.train.patch.training.learning_rate', train.learning_rate, {type: 'number', min: 0, step: 'any'}) + '</div>';
      body += '<div class="field-row">' + selectField('Precision', 'config.train.patch.training.precision', train.precision, [['bf16', 'bfloat16'], ['fp32', 'float32']]) + selectField('Gradient checkpointing', 'config.train.patch.training.gradient_checkpointing', train.gradient_checkpointing, [['true', 'Enabled'], ['false', 'Disabled']]) + '</div>';
      if (doc.dataset.partition && tasks.length) {
        body += `<div class="field"><span class="field-label">Training task groups <button class="reset-field" data-reset="config.train.patch.dataset.partition.groups">Reset override</button></span><small>${doc.dataset.partition.groups?.length ?? tasks.length} / ${tasks.length} selected · ${esc(suite)}</small></div><div class="task-shortcuts"><button data-task-action="all">All</button><button data-task-action="first-two">First 2</button><button data-task-action="clear">Clear</button></div><div class="task-options">${tasks.map((task, index) => `<label class="task-option"><input type="checkbox" data-train-task="${esc(task)}" ${!doc.dataset.partition.groups || doc.dataset.partition.groups.includes(task) ? 'checked' : ''}><span>${index} · ${esc(task.replace(/_/g, ' '))}</span></label>`).join('')}</div>`;
        body += '<div class="field-row">' + field('Episodes per group', 'config.train.patch.dataset.partition.max_episodes_per_group', doc.dataset.partition.max_episodes_per_group, {type: 'number', min: 1, integer: true}) + field('Frames per episode', 'config.train.patch.dataset.partition.max_frames_per_episode', doc.dataset.partition.max_frames_per_episode, {type: 'number', min: 1, integer: true}) + '</div>';
      }
      body += '<p class="helper">The plugin declares supported training implementations. GPU and dataset availability are checked by the CLI.</p>';
    }
    return `<details open data-section="train"><summary><span class="pill train">T</span>Training configuration<span class="section-chip">${esc(item.state.train?.profileSource !== item.path && item.state.train?.profile ? 'inherited' : 'local')}</span></summary><div class="details-body">${body}</div></details>`;
  }
  function renderEvaluation(item) {
    const resolved = C.resolve(item, 'evaluate', catalog), local = item.node.config?.evaluate, inherited = item.parent !== null ? C.flatten(workflow).find(node => node.path === item.parent).state.evaluate : undefined;
    let body = profileField(item, 'evaluate');
    if (resolved && !resolved.unavailable) {
      const {document: doc, components} = resolved, policy = components.policy, method = C.family(resolved, 'evaluate');
      body += `<div class="profile-info"><strong>${esc(methodLabel(method))}</strong> · ${esc(policy?.settings?.checkpoint_id || 'mock policy')}<br>${esc(doc.deployment ? doc.deployment.profile + ' deployment · ' + doc.deployment.renderer : 'Native mock evaluation')}</div>`;
      body += '<div class="field"><label for="checkpoint-source">Checkpoint source</label><select id="checkpoint-source" data-checkpoint><option value="__inherit__" ' + (!own(local, 'checkpoint_from') ? 'selected' : '') + '>' + esc(own(inherited, 'checkpoint_from') ? 'Inherit · ' + (inherited.checkpoint_from || '<root>') + ':train' : 'Configured policy checkpoint') + '</option>';
      body += analysis.entries.filter(node => (node.node.execute || []).includes('train')).map(source => {
        const sourceMethod = C.family(C.resolve(source, 'train', catalog), 'train');
        const compatible = C.bindingCompatible(C.resolve(source, 'train', catalog), resolved);
        return `<option value="${esc('node:' + source.path)}" ${own(local, 'checkpoint_from') && local.checkpoint_from === source.path ? 'selected' : ''} ${compatible ? '' : 'disabled'}>${esc((source.path || '<root>') + ':train · ' + methodLabel(sourceMethod) + (compatible ? '' : ' · incompatible'))}</option>`;
      }).join('');
      if (own(local, 'checkpoint_from') && !analysis.entries.some(node => node.path === local.checkpoint_from && (node.node.execute || []).includes('train'))) body += `<option value="${esc('node:' + local.checkpoint_from)}" selected>Missing: ${esc(local.checkpoint_from || '<root>')}</option>`;
      body += '</select><small>Training sources add a dependency and use their finalized output. Configured checkpoints use the policy settings.</small></div>';
      if (own(inherited, 'checkpoint_from') && !own(local, 'checkpoint_from')) body += `<p class="inheritance-note">Binding inherited from <button class="text-button" data-select="${esc(item.state.evaluate.checkpointSource)}">${esc(item.state.evaluate.checkpointSource || '<root>')}</button>. Edit that node to remove the binding, or choose another training source here.</p>`;
      const quantization = policy?.settings?.runtime?.quantization;
      if (quantization !== undefined) body += selectField('Inference quantization', 'config.evaluate.component_patches.policy.settings.runtime.quantization', quantization, (resolved.plugin?.deployment?.quantization || [quantization]).map(value => [value, value === 'none' ? 'Full precision' : value]));
      const benchmark = components.benchmark, protocol = components.protocol;
      if (benchmark?.type === 'libero') body += field('Evaluation task indices', 'config.evaluate.component_patches.benchmark.settings.task_indices', benchmark.settings.task_indices, {hint: 'Comma-separated task indices, e.g. 0, 1. Use all for the entire suite.'}) + '<div class="task-shortcuts"><button data-eval-tasks="0">Task 0</button><button data-eval-tasks="0,1">First 2</button><button data-eval-tasks="all">All tasks</button></div>';
      body += '<div class="field-row">' + field('Episodes per task', 'config.evaluate.component_patches.protocol.execution.rollouts_per_task', protocol?.execution?.rollouts_per_task, {type: 'number', min: 1, step: 1, integer: true}) + field('Max. episode steps', 'config.evaluate.component_patches.protocol.execution.maximum_episode_steps', protocol?.execution?.maximum_episode_steps, {type: 'number', min: 1, step: 1, integer: true}) + '</div>';
      if (doc.deployment) body += selectField('Renderer', 'config.evaluate.patch.deployment.renderer', doc.deployment.renderer, [['egl', 'EGL · headless'], ['glfw', 'GLFW · display']]);
      for (const kind of ['benchmark', 'policy', 'protocol', 'metrics', 'artifacts']) {
        const current = C.get(item.node, 'config.evaluate.patch.components.' + kind);
        let candidates = catalog.components[kind] || [];
        if (kind === 'policy' && policy) candidates = candidates.filter(option => option.document.type === policy.type);
        if (kind === 'benchmark' && benchmark) candidates = candidates.filter(option => option.document.type === benchmark.type);
        body += `<div class="field"><label for="component-${kind}">${esc(kind[0].toUpperCase() + kind.slice(1))} component</label><select id="component-${kind}" data-component="${kind}"><option value="__inherit__" ${!current ? 'selected' : ''}>From profile / parent</option>${options(candidates, current, option => option.ref)}</select><small class="effective">${esc(doc.components[kind])}</small></div>`;
      }
      const identity = resolved.plugin?.capabilities?.checkpoint_binding?.adapter_identity;
      if (identity && !own(item.state.evaluate, 'checkpoint_from')) {
        body += '<div class="profile-info">Select a training checkpoint above, or configure an existing adapter and its hashes below.</div>';
        for (const key of [identity.path, ...identity.hashes]) body += field(key.replace(/_/g, ' '), 'config.evaluate.component_patches.policy.settings.' + identity.settings_path + '.' + key, C.get(policy.settings, identity.settings_path + '.' + key));
      }
    }
    return `<details open data-section="evaluate"><summary><span class="pill evaluate">E</span>Evaluation configuration<span class="section-chip">${esc(item.state.evaluate?.profileSource !== item.path && item.state.evaluate?.profile ? 'inherited' : 'local')}</span></summary><div class="details-body">${body}</div></details>`;
  }
  function renderInspector() {
    const open = new Map([...document.querySelectorAll('[data-section]')].map(element => [element.dataset.section, element.open]));
    const item = entry(), node = item.node;
    let html = `<div class="inspector-header" data-inspector-key="header"><div><h2>${item.path ? 'Node settings' : 'Experiment settings'}</h2><p>${esc(item.path || '<root>')}</p></div><div class="node-actions"><button class="button icon-button" data-action="duplicate" aria-label="Duplicate node" title="Duplicate subtree" ${!item.path ? 'disabled' : ''}>${icon('copy')}</button><button class="button icon-button" data-action="delete" aria-label="Delete node" title="Delete subtree" ${!item.path ? 'disabled' : ''}>${icon('trash')}</button></div></div>`;
    const checks = analysis.issues.filter(issue => issue.path === item.path && issue.severity !== 'info');
    if (checks.length) html += `<section class="node-checks" data-inspector-key="checks" aria-label="Selected node checks">${checks.map(issue => `<p class="${issue.severity}">${esc(issue.message)}${issue.fix ? `<button class="text-button" data-fix="${esc(issue.fix)}" data-fix-path="${esc(issue.path)}">Evaluate only children</button>` : ''}</p>`).join('')}</section>`;
    html += '<section class="inspector-section" data-inspector-key="identity">' + field('Display name', 'name', node.id, {reset: !!item.path}) + `<div class="field"><label for="node-id">Node ID</label><input id="node-id" data-rename value="${esc(node.id)}" maxlength="120"><small>Stable path segment. References update when you rename a node.</small></div>`;
    if (item.parent !== null) html += `<div class="field"><label for="node-parent">Parent node</label><select id="node-parent" data-parent>${options(analysis.entries.filter(other => other.path !== item.path && !other.path.startsWith(item.path + '/')).map(other => ({ref: other.path, label: other.path || '<root>'})), item.parent)}</select></div><div class="button-row"><button class="button small" data-action="up">↑ Move up</button><button class="button small" data-action="down">↓ Move down</button></div>`;
    else html += field('Concurrent workers', 'execution.max_workers', 2, {type: 'number', integer: true, min: 1, step: 1, hint: '1 = sequential. Maximum 32. Workers do not allocate GPUs.'});
    html += '</section><section class="inspector-section" data-inspector-key="execution"><h3>Execute at this node</h3><div class="execution-switches">' + C.PHASES.map(phase => `<label class="phase-toggle"><input type="checkbox" data-execute="${phase}" ${(node.execute || []).includes(phase) ? 'checked' : ''}>${phase === 'train' ? 'Train' : 'Evaluate'}</label>`).join('') + `</div><p class="execution-hint">${(node.execute || []).length ? 'Enabled phases create runs at this node, even when children execute too.' : 'Group only. Settings below are inherited by children without creating a run here.'}</p></section>`;
    html += renderTraining(item) + renderEvaluation(item);
    html += '<details data-section="dependencies"><summary>Additional dependencies</summary><div class="details-body"><p class="helper">Checkpoint handoff already adds its training dependency. Explicit dependencies affect every phase executed at this node.</p>';
    html += (node.depends_on || []).map((dependency, index) => `<div class="dependency-row"><span>${esc((dependency.node || '<root>') + ':' + dependency.phase)}</span><button data-remove-dependency="${index}" aria-label="Remove dependency">×</button></div>`).join('');
    html += `<div class="dependency-add"><select id="dependency-target" aria-label="Dependency target"><option value="">Select a phase…</option>${analysis.entries.flatMap(other => (other.node.execute || []).filter(phase => other.path !== item.path || !(node.execute || []).includes(phase)).map(phase => ({ref: JSON.stringify([other.path, phase]), label: (other.path || '<root>') + ':' + phase}))).map(option => `<option value="${esc(option.ref)}">${esc(option.label)}</option>`).join('')}</select><button class="button small" data-action="add-dependency">Add</button></div></div></details>`;
    const draft = advancedDraft?.path === selected ? advancedDraft.text : JSON.stringify(node.config || {}, null, 2);
    html += '<details data-section="advanced"><summary>Advanced phase configuration</summary><div class="details-body"><p class="helper">Edit local patches and component overrides as JSON. Apply to update the workflow. Unapplied edits are saved in this browser and stay attached to this node.</p><div class="field"><label for="advanced-config">Local node.config</label><textarea id="advanced-config" class="advanced-editor" spellcheck="false">' + esc(draft) + '</textarea></div><div id="advanced-error" class="inline-error" role="alert" ' + (!advancedError || advancedDraft?.path !== selected ? 'hidden' : '') + '>' + esc(advancedError) + '</div><div class="button-row"><button class="button small primary" data-action="apply-advanced">Apply local configuration</button><button class="button small" data-action="discard-advanced" hidden>Discard edits</button></div></div></details>';
    html += '<section class="inspector-section"><button class="button add-outline" data-action="add-child">' + icon('plus') + 'Add child node</button></section>';
    window.OVLABStudioDOM.render($('#inspector'), html);
    for (const element of document.querySelectorAll('[data-section]')) if (open.has(element.dataset.section)) element.open = open.get(element.dataset.section);
  }
  function renderPlan() {
    const errors = analysis.issues.filter(issue => issue.severity === 'error');
    const workers = workflow.execution?.max_workers || 2;
    const rows = [...analysis.rows, ...(analysis.blockedRows || [])];
    const header = `<div class="section-bar"><div><h2>Execution plan</h2><p>${analysis.phaseCount} phases · ${workers === 1 ? 'Sequential execution' : `Up to ${workers} concurrent phases`} · dependency order</p></div><span class="pill ${errors.length ? 'train' : 'bound'}">${errors.length ? errors.length + ' errors to resolve' : 'Browser checks passed'}</span></div>`;
    const tableRows = rows.map((row, index) => {
      let settings = 'Select or repair the phase profile.';
      if (row.resolved && !row.resolved.unavailable) {
        if (row.phase === 'train') {
          const profile = row.resolved.document;
          settings = `${profile.training.max_steps} optimizer steps · batch ${profile.training.per_device_batch_size} · ${profile.training.precision}`;
          settings += `<small>${esc(profile.dataset.ref)}${profile.dataset.partition?.groups ? ' · ' + profile.dataset.partition.groups.length + ' task groups' : ''}</small>`;
        } else {
          const components = row.resolved.components, taskSet = components.benchmark?.settings?.task_indices;
          const tasks = Array.isArray(taskSet) ? taskSet.join(', ') : taskSet || 'all';
          const precision = components.policy?.settings.runtime?.quantization || 'native';
          settings = `${esc(components.benchmark?.settings?.suite || 'mock')} · tasks ${esc(tasks)}<small>${esc(components.protocol?.execution?.rollouts_per_task)} episode(s) · ${esc(components.protocol?.execution?.maximum_episode_steps)} max. steps · ${esc(precision === 'none' ? 'full precision' : precision)}</small>`;
        }
      }
      const attention = row.blocked || errors.some(issue => issue.path === row.path);
      return `<tr class="${attention ? 'plan-attention' : ''}"><td>${row.blocked ? '—' : String(index + 1).padStart(2, '0')}</td><td><button data-select="${esc(row.path)}">${esc(row.path || '<root>')}</button><small title="${esc(row.resolved?.source?.ref || row.resolved?.reference)}">${esc((row.resolved?.source?.ref || row.resolved?.reference || 'Profile missing').split('/').at(-1))}</small>${row.blocked ? '<span class="pill train">Blocked by dependencies</span>' : ''}</td><td><span class="pill ${row.phase}">${row.phase === 'train' ? 'Train' : 'Evaluate'}</span></td><td>${settings}</td><td>${row.checkpoint !== undefined ? '<span class="pill bound">' + esc((row.checkpoint || '<root>') + ':train') + '</span>' : '<span class="muted">' + (row.phase === 'train' ? 'Final training output' : 'Policy checkpoint') + '</span>'}${row.dependencies.size ? `<small>After: ${[...row.dependencies].map(key => { const [path, phase] = JSON.parse(key); return esc((path || '<root>') + ':' + phase); }).join(', ')}</small>` : ''}</td></tr>`;
    }).join('');
    const table = rows.length ? `<div class="plan-scroll"><table class="plan-table"><thead><tr><th scope="col">#</th><th scope="col">Node / profile</th><th scope="col">Phase</th><th scope="col">Settings</th><th scope="col">Checkpoint / ordering</th></tr></thead><tbody>${tableRows}</tbody></table></div>` : '<div class="empty-state"><strong>No phases configured yet</strong><p>Add a training or evaluation node, or enable a phase in the inspector.</p><button class="button primary small" data-action="add-child">Add a node</button></div>';
    window.OVLABStudioDOM.render($('#view-plan'), header + (errors.length ? '<p class="plan-warning">Preview only. Resolve the configuration errors before this workflow can run. Blocked phases remain visible below.</p>' : '') + table + `<div class="command-box">Export YAML into <code>experiments/workflows/</code>, then validate with the actual resolver:<br><code>ovlab run experiments/workflows/${esc(workflow.id)}.yaml --plan</code><br>Model, dataset and GPU availability are checked when the experiment starts.</div>`);
  }
  function renderIssues() {
    const errors = analysis.issues.filter(issue => issue.severity === 'error'), warnings = analysis.issues.filter(issue => issue.severity === 'warning');
    const status = $('#validation-status'); status.className = 'validation-status' + (errors.length ? ' invalid' : warnings.length ? ' warning' : '');
    status.innerHTML = errors.length ? `${errors.length} ${errors.length === 1 ? 'error' : 'errors'}` : warnings.length ? `${warnings.length} ${warnings.length === 1 ? 'warning' : 'warnings'}` : `${icon('check')}Browser checks passed`;
    status.setAttribute('aria-label', errors.length ? `${errors.length} configuration errors. Show checks.` : warnings.length ? `${warnings.length} configuration warnings. Show checks.` : 'Browser checks passed. Show checks.');
    window.OVLABStudioDOM.render($('#issues'), analysis.issues.map(issue => `<div class="issue-row ${issue.severity}"><span class="issue-icon" aria-hidden="true">${issue.severity === 'error' ? '!' : issue.severity === 'warning' ? '!' : 'i'}</span><span><button class="issue-node" data-select="${esc(issue.path)}">${esc(issue.path || '<root>')}</button> · ${esc(issue.message)}${issue.fix ? `<button class="text-button issue-fix" data-fix="${esc(issue.fix)}" data-fix-path="${esc(issue.path)}" ${hasDraft() ? 'disabled' : ''}>Evaluate only children</button>` : ''}</span></div>`).join(''));
    $('#export').disabled = !!errors.length || hasDraft();
    $('#export').title = errors.length ? 'Resolve errors before exporting.' : hasDraft() ? 'Apply or discard your editor draft first.' : 'Download a CLI-compatible workflow YAML';
  }
  function renderDraftState() {
    const pending = hasDraft(), owned = advancedDraft?.path === selected;
    $('#draft-notice').hidden = !pending;
    window.OVLABStudioDOM.render($('#draft-notice'), pending ? `<div><strong>${sourceDirty ? 'Unapplied YAML changes' : 'Unapplied local configuration'}</strong><span>${sourceDirty ? 'Your source draft is saved. Apply or discard it before editing nodes or exporting.' : 'Draft saved for ' + esc(advancedDraft.path || '<root>') + '. Apply or discard it to continue editing.'}</span></div><div class="button-row"><button class="button small" data-draft-action="open">Open draft</button><button class="button small" data-draft-action="discard">Discard edits</button></div>` : '');
    for (const control of document.querySelectorAll('#inspector [data-field], #inspector [data-rename], #inspector [data-parent], #inspector [data-execute], #inspector [data-profile], #inspector [data-component], #inspector [data-checkpoint], #inspector [data-select-field], #inspector [data-train-task], #inspector [data-reset], #inspector [data-task-action], #inspector [data-eval-tasks], #inspector [data-remove-dependency], #dependency-target')) control.disabled = pending;
    for (const button of document.querySelectorAll('#inspector [data-action]')) {
      if (button.dataset.action === 'apply-advanced') button.disabled = !owned || sourceDirty;
      else if (button.dataset.action === 'discard-advanced') { button.hidden = !owned; button.disabled = false; }
      else button.disabled = pending || (!selected && ['duplicate', 'delete', 'up', 'down'].includes(button.dataset.action));
    }
    for (const button of document.querySelectorAll('[data-fix]')) button.disabled = pending;
    $('#add-node').disabled = pending;
    $('#undo').disabled = pending || !history.length;
    $('#redo').disabled = pending || !future.length;
    $('#undo').title = pending ? 'Apply or discard the draft before undoing.' : 'Undo (Ctrl/⌘ Z)';
    $('#redo').title = pending ? 'Apply or discard the draft before redoing.' : 'Redo (Ctrl/⌘ Shift Z)';
    $('#yaml-source').readOnly = !!advancedDraft;
    $('#apply-yaml').disabled = !sourceDirty || !!advancedDraft;
    $('#discard-yaml').hidden = !sourceDirty;
    $('#source-status').textContent = sourceDirty ? 'Draft saved locally. Apply to update the workflow.' : advancedDraft ? 'Apply or discard the local configuration draft before editing source.' : 'Applied workflow. Edit and apply to update the builder.';
    const advanced = $('#advanced-config');
    if (advanced) advanced.readOnly = sourceDirty || (!!advancedDraft && !owned);
    renderIssues();
  }
  function discardDraft() {
    sourceDirty = false; advancedDraft = null; advancedError = '';
    $('#source-error').hidden = true; render(); persist(); toast('Unapplied edits discarded.');
  }
  function openDraft() {
    if (sourceDirty) { view = 'yaml'; render(); $('#yaml-source').focus(); }
    else if (advancedDraft) {
      select(advancedDraft.path);
      const details = document.querySelector('[data-section="advanced"]'); details.open = true;
      $('#advanced-config').focus(); $('#advanced-config').scrollIntoView({block: 'center'});
    }
  }
  function render() {
    analysis = C.analyze(workflow, catalog);
    if (!analysis.entries.length) throw new Error('Invalid workflow structure. Import a valid workflow to continue.');
    if (!analysis.entries.some(item => item.path === selected)) selected = '';
    $('#workflow-title').textContent = workflow.name;
    $('#workflow-title').title = workflow.name;
    $('#workflow-subtitle').textContent = workflow.id + ' · ' + (workflow.execution?.max_workers === 1 ? 'Sequential execution' : 'Parallel execution');
    const inventory = [...analysis.rows, ...(analysis.blockedRows || [])];
    const trains = inventory.filter(row => row.phase === 'train').length, evaluations = inventory.filter(row => row.phase === 'evaluate').length;
    $('#workflow-counts').innerHTML = `<span class="metric"><strong>${trains}</strong>training</span><span class="metric"><strong>${evaluations}</strong>evaluation</span>`;
    $('#plan-count').textContent = analysis.phaseCount;
    $('#undo').disabled = !history.length; $('#redo').disabled = !future.length;
    $('#copy-command').textContent = `ovlab run experiments/workflows/${workflow.id}.yaml --plan`;
    renderOutline(); renderCanvas(); renderInspector(); renderPlan(); renderDraftState();
    if (!sourceDirty) $('#yaml-source').value = C.dumpYaml(workflow);
    for (const button of document.querySelectorAll('[data-view]')) { button.setAttribute('aria-selected', String(button.dataset.view === view)); button.tabIndex = button.dataset.view === view ? 0 : -1; }
    for (const name of ['builder', 'plan', 'yaml']) $('#view-' + name).hidden = name !== view;
  }
  function revealSelected() {
    if (view !== 'builder') return;
    const area = $('#canvas-scroll');
    const card = [...document.querySelectorAll('.node-card')].find(element => element.dataset.select === selected);
    if (!card) return;
    const item = card.getBoundingClientRect(), viewport = area.getBoundingClientRect(), margin = 16;
    const dx = item.left < viewport.left + margin ? item.left - viewport.left - margin : item.right > viewport.right - margin ? item.right - viewport.right + margin : 0;
    const dy = item.top < viewport.top + margin ? item.top - viewport.top - margin : item.bottom > viewport.bottom - margin ? item.bottom - viewport.bottom + margin : 0;
    area.scrollBy(dx, dy);
  }
  function select(path) { const previous = selected; selected = path; render(); persist(); if (previous !== selected) $('.inspector').scrollTop = 0; revealSelected(); }
  function showAdd() {
    if (!canChange()) return;
    const inherited = C.resolve(entry(), 'evaluate', catalog);
    const trained = [...analysis.entries].filter(item => (item.path === selected || selected.startsWith(item.path ? item.path + '/' : '')) && (item.node.execute || []).includes('train')).at(-1);
    const trainedConfig = trained && C.resolve(trained, 'train', catalog);
    const compatible = trainedConfig && C.bindingCompatible(trainedConfig, evaluationFor(trainedConfig)?.resolved);
    $('#add-parent').textContent = 'Parent: ' + (selected || '<root>') + (inherited && !inherited.unavailable ? ' · A new evaluation inherits this branch’s settings.' : compatible ? ' · A new evaluation can use this branch’s training checkpoint.' : ' · Choose a node type, then configure its plugin profile.');
    $('[data-template="handoff"] span:last-child').textContent = 'Use a compatible plugin training and evaluation configuration with a finalized checkpoint binding.';
    $('#add-dialog').showModal();
  }
  function addNode(template) {
    change(() => {
      const inheritedTraining = C.resolve(entry(), 'train', catalog);
      const referenceTraining = referenceConfigurations('train')[0];
      const training = inheritedTraining && !inheritedTraining.unavailable ? inheritedTraining : referenceTraining?.resolved;
      const parent = entry().node, id = C.uniqueId(parent, template === 'handoff' ? 'train-evaluate' : template === 'evaluate' ? 'evaluation' : template === 'train' ? 'training' : 'group');
      const path = selected ? selected + '/' + id : id, node = {id, name: template === 'group' ? 'New group' : template === 'handoff' ? 'Train → evaluate' : template === 'train' ? 'Training' : 'Evaluation'};
      if (template === 'train' || template === 'handoff') {
        node.execute = ['train'];
        node.config = inheritedTraining && !inheritedTraining.unavailable ? {} : referenceTraining ? {train: C.clone(referenceTraining.config)} : {};
      }
      if (template === 'evaluate') {
        node.execute = ['evaluate']; node.config = {};
        const inherited = C.resolve(entry(), 'evaluate', catalog);
        if (!inherited || inherited.unavailable) {
          const source = [...analysis.entries].filter(item => (item.path === selected || (selected && (!item.path || selected.startsWith(item.path + '/')))) && (item.node.execute || []).includes('train')).at(-1);
          const trained = source && C.resolve(source, 'train', catalog), choice = evaluationFor(trained);
          if (choice) {
            node.config.evaluate = C.clone(choice.config);
            delete node.config.evaluate.checkpoint_from;
            if (source && C.bindingCompatible(trained, choice.resolved)) node.config.evaluate.checkpoint_from = source.path;
          }
        }
      }
      if (template === 'handoff') {
        const choice = evaluationFor(training);
        const evaluate = choice ? C.clone(choice.config) : {};
        delete evaluate.checkpoint_from;
        if (choice && C.bindingCompatible(training, choice.resolved)) evaluate.checkpoint_from = path;
        node.children = [{id: 'evaluate-checkpoint', name: 'Evaluate finalized checkpoint', execute: ['evaluate'], config: {evaluate}}];
      }
      (parent.children ||= []).push(node); selected = path;
    });
    $('#add-dialog').close();
    requestAnimationFrame(revealSelected);
  }
  async function nodeAction(action) {
    const item = entry(), node = item.node;
    if (action === 'add-child') showAdd();
    if (action === 'duplicate' && item.path) change(() => {
      const parent = analysis.entries.find(other => other.path === item.parent).node, copy = C.clone(node);
      copy.id = C.uniqueId(parent, node.id + '-copy'); copy.name = (node.name || node.id) + ' (copy)';
      const path = item.parent ? item.parent + '/' + copy.id : copy.id;
      C.rewriteReferences(copy, item.path, path); parent.children.push(copy); selected = path;
    });
    if (action === 'delete' && item.path && await confirmAction('Delete this node?', `Delete ${item.path} and its children? References to deleted nodes will be marked as errors. Undo can restore the subtree.`, 'Delete subtree')) change(() => {
      const parent = analysis.entries.find(other => other.path === item.parent).node; parent.children.splice(parent.children.indexOf(node), 1); selected = item.parent;
    });
    if (['up', 'down'].includes(action) && item.path) change(() => {
      const children = analysis.entries.find(other => other.path === item.parent).node.children, old = children.indexOf(node), next = old + (action === 'up' ? -1 : 1);
      if (next >= 0 && next < children.length) [children[old], children[next]] = [children[next], children[old]];
    });
    if (action === 'add-dependency') {
      const value = $('#dependency-target').value; if (!value) return;
      const [path, phase] = JSON.parse(value);
      if (!(node.depends_on || []).some(dependency => dependency.node === path && dependency.phase === phase)) change(() => { (node.depends_on ||= []).push({node: path, phase}); });
    }
    if (action === 'apply-advanced') {
      try {
        if (!advancedDraft || advancedDraft.path !== selected || sourceDirty) return;
        const value = C.parse($('#advanced-config').value);
        if (!C.object(value)) throw new Error('Local configuration must be an object.');
        const candidate = C.clone(workflow), other = C.flatten(candidate).find(other => other.path === item.path); other.node.config = value;
        const result = C.analyze(candidate, catalog); if (!result.entries.length) throw new Error(result.issues.find(issue => issue.severity === 'error').message);
        if (change(() => { node.config = value; }, 'advanced')) toast('Local configuration applied.');
      } catch (error) {
        advancedError = error.message; $('#advanced-error').textContent = advancedError; $('#advanced-error').hidden = false;
      }
    }
    if (action === 'discard-advanced') discardDraft();
  }
  document.addEventListener('click', event => {
    const draft = event.target.closest('[data-draft-action]'); if (draft) { draft.dataset.draftAction === 'open' ? openDraft() : discardDraft(); return; }
    const fix = event.target.closest('[data-fix]'); if (fix) { change(() => { const node = analysis.entries.find(item => item.path === fix.dataset.fixPath).node; node.execute = (node.execute || []).filter(phase => phase !== 'evaluate'); }); return; }
    const shortcut = event.target.closest('[data-task-action], [data-eval-tasks]');
    if (shortcut) {
      change(() => {
        if (shortcut.hasAttribute('data-eval-tasks')) C.set(entry().node, 'config.evaluate.component_patches.benchmark.settings.task_indices', shortcut.dataset.evalTasks === 'all' ? 'all' : shortcut.dataset.evalTasks.split(',').map(Number));
        else {
          const resolved = C.resolve(entry(), 'train', catalog), suite = resolved.document.dataset.ref.split('/').at(-1), tasks = catalog.tasks[suite] || [];
          C.set(entry().node, 'config.train.patch.dataset.partition.groups', shortcut.dataset.taskAction === 'all' ? tasks : shortcut.dataset.taskAction === 'first-two' ? tasks.slice(0, 2) : []);
        }
      }); return;
    }
    const selectedButton = event.target.closest('[data-select]'); if (selectedButton) { select(selectedButton.dataset.select); return; }
    const action = event.target.closest('[data-action]'); if (action) { nodeAction(action.dataset.action); return; }
    const reset = event.target.closest('[data-reset]'); if (reset) { event.preventDefault(); change(() => C.set(entry().node, reset.dataset.reset, undefined)); return; }
    const remove = event.target.closest('[data-remove-dependency]'); if (remove) change(() => entry().node.depends_on.splice(Number(remove.dataset.removeDependency), 1));
  });
  $('#inspector').addEventListener('change', event => {
    const target = event.target, node = entry().node;
    if (target.matches('[data-rename]')) { change(() => { selected = C.renameNode(workflow, selected, target.value.trim()); }); return; }
    if (target.matches('[data-parent]')) { change(() => { selected = C.moveNode(workflow, selected, target.value); }); return; }
    if (target.matches('[data-execute]')) { change(() => { const phases = new Set(node.execute || []); target.checked ? phases.add(target.dataset.execute) : phases.delete(target.dataset.execute); node.execute = C.PHASES.filter(phase => phases.has(phase)); }); return; }
    if (target.matches('[data-profile]')) { change(() => C.set(node, 'config.' + target.dataset.profile + '.profile', target.value === '__inherit__' ? undefined : target.value)); return; }
    if (target.matches('[data-component]')) { change(() => C.set(node, 'config.evaluate.patch.components.' + target.dataset.component, target.value === '__inherit__' ? undefined : target.value)); return; }
    if (target.matches('[data-checkpoint]')) { change(() => C.set(node, 'config.evaluate.checkpoint_from', target.value === '__inherit__' ? undefined : target.value.slice(5))); return; }
    if (target.matches('[data-select-field]')) { change(() => C.set(node, target.dataset.selectField, target.value === '__inherit__' ? undefined : target.value === 'true' ? true : target.value === 'false' ? false : target.value)); return; }
    if (target.matches('[data-train-task]')) {
      change(() => {
        const resolved = C.resolve(entry(), 'train', catalog), suite = resolved.document.dataset.ref.split('/').at(-1);
        const groups = new Set(resolved.document.dataset.partition.groups || catalog.tasks[suite]);
        target.checked ? groups.add(target.dataset.trainTask) : groups.delete(target.dataset.trainTask);
        C.set(node, 'config.train.patch.dataset.partition.groups', (catalog.tasks[suite] || []).filter(task => groups.has(task)));
      }); return;
    }
    if (target.matches('[data-field]')) {
      let value = target.value.trim();
      if (value === '') value = undefined;
      else if (target.dataset.field.endsWith('.task_indices')) {
        if (value !== 'all') { const values = value.split(',').map(value => value.trim()); if (values.some(value => !/^\d+$/.test(value))) { toast('Use comma-separated integer task indices or all.'); return; } value = values.map(Number); }
      } else if (['integer', 'number'].includes(target.dataset.valueType)) {
        value = Number(value); if (!Number.isFinite(value) || (target.dataset.valueType === 'integer' && !Number.isInteger(value))) { toast('Enter a finite ' + target.dataset.valueType + '.'); return; }
      }
      change(() => C.set(node, target.dataset.field, value));
    }
  });
  $('#inspector').addEventListener('input', event => {
    if (event.target.id !== 'advanced-config') return;
    const text = event.target.value;
    advancedDraft = text === JSON.stringify(entry().node.config || {}, null, 2) ? null : {path: selected, text};
    advancedError = ''; $('#advanced-error').hidden = true;
    renderDraftState(); persist();
  });
  $('#outline').addEventListener('dragstart', event => { const row = event.target.closest('[draggable]'); if (!row) return; event.dataTransfer.setData('text/plain', row.dataset.select); event.dataTransfer.effectAllowed = 'move'; });
  $('#outline').addEventListener('dragover', event => { const row = event.target.closest('[data-drop]'); if (row) { event.preventDefault(); row.classList.add('drag-over'); } });
  $('#outline').addEventListener('dragleave', event => event.target.closest('[data-drop]')?.classList.remove('drag-over'));
  $('#outline').addEventListener('drop', event => { event.preventDefault(); const row = event.target.closest('[data-drop]'), path = event.dataTransfer.getData('text/plain'); if (row && path) change(() => { selected = C.moveNode(workflow, path, row.dataset.drop); }); });
  $('#outline').addEventListener('keydown', event => {
    const row = event.target.closest('[data-select]');
    if (!row || !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const current = analysis.entries.findIndex(item => item.path === row.dataset.select), item = analysis.entries[current];
    let next = event.key === 'Home' ? 0 : event.key === 'End' ? analysis.entries.length - 1 : event.key === 'ArrowUp' ? Math.max(0, current - 1) : event.key === 'ArrowDown' ? Math.min(analysis.entries.length - 1, current + 1) : current;
    if (event.key === 'ArrowLeft' && item.parent !== null) next = analysis.entries.findIndex(other => other.path === item.parent);
    if (event.key === 'ArrowRight' && item.node.children?.length) next = current + 1;
    const path = analysis.entries[next].path; select(path);
    [...document.querySelectorAll('#outline [data-select]')].find(element => element.dataset.select === path)?.focus();
  });
  $('#add-node').addEventListener('click', showAdd);
  for (const button of document.querySelectorAll('[data-template]')) button.addEventListener('click', () => addNode(button.dataset.template));
  $('#undo').addEventListener('click', () => undo()); $('#redo').addEventListener('click', () => undo(true));
  for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => {
    view = button.dataset.view; render();
  });
  $('.workspace-tabs').addEventListener('keydown', event => {
    if (!event.target.matches('[role=tab]') || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); const tabs = [...document.querySelectorAll('[role=tab]')], current = tabs.indexOf(event.target);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].click(); tabs[next].focus();
  });
  $('#yaml-source').addEventListener('input', () => { sourceDirty = $('#yaml-source').value !== C.dumpYaml(workflow); $('#source-error').hidden = true; renderDraftState(); persist(); });
  $('#discard-yaml').addEventListener('click', discardDraft);
  $('#apply-yaml').addEventListener('click', () => {
    try {
      const document = C.parse($('#yaml-source').value), result = C.analyze(document, catalog);
      if (!result.entries.length) throw new Error(result.issues.find(item => item.severity === 'error')?.message || 'Import a workflow document.');
      if (change(() => { workflow = document; }, 'source')) { $('#source-error').hidden = true; toast('Workflow source applied.'); }
    } catch (error) { $('#source-error').textContent = error.message; $('#source-error').hidden = false; }
  });
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied to clipboard.'); }
    catch (_) { const area = document.createElement('textarea'); area.value = text; area.style.position = 'fixed'; area.style.opacity = '0'; document.body.append(area); area.select(); const copied = document.execCommand('copy'); area.remove(); toast(copied ? 'Copied to clipboard.' : 'Clipboard unavailable. Select the YAML source and copy it manually.'); }
  }
  $('#copy-yaml').addEventListener('click', () => copy($('#yaml-source').value));
  $('#copy-command').addEventListener('click', () => copy($('#copy-command').textContent));
  $('#export').addEventListener('click', () => {
    if ($('#export').disabled) return;
    const blob = new Blob([C.dumpYaml(workflow)], {type: 'application/yaml;charset=utf-8'}), url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = workflow.id + '.yaml'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('YAML exported. Validate it with ovlab run … --plan.');
  });
  $('#import').addEventListener('click', () => $('#import-file').click());
  $('#import-file').addEventListener('change', async event => {
    try { const file = event.target.files[0]; if (file) { if (file.size > 2000000) throw new Error('Configurations must be smaller than 2 MB.'); await replaceWorkflow(C.parse(await file.text()), 'Workflow imported.'); } }
    catch (error) { toast(error.message); } finally { event.target.value = ''; }
  });
  $('#new-workflow').addEventListener('click', () => replaceWorkflow(blank(), 'New workflow created.').catch(error => toast(error.message)));
  $('#preset').innerHTML += options(catalog.workflows, '', item => item.ref.split('/').at(-1));
  $('#preset').addEventListener('change', async event => { const preset = catalog.workflows.find(item => item.ref === event.target.value); if (preset) await replaceWorkflow(preset.document, 'Repository workflow loaded.'); event.target.value = ''; });
  $('#validation-status').addEventListener('click', () => { $('#issues').scrollIntoView({block: 'nearest'}); if (!analysis.issues.length) toast('Browser checks passed. Validate the export with ovlab run … --plan.'); });
  function changeZoom(delta) {
    const area = $('#canvas-scroll'), centerX = (area.scrollLeft + area.clientWidth / 2) / zoom, centerY = (area.scrollTop + area.clientHeight / 2) / zoom;
    zoom = Math.min(1.5, Math.max(0.35, Math.round((zoom + delta) * 100) / 100)); renderCanvas();
    area.scrollTo(centerX * zoom - area.clientWidth / 2, centerY * zoom - area.clientHeight / 2);
  }
  function fitCanvas(overview = true) {
    if (view !== 'builder') return;
    const graph = $('#canvas .graph'), area = $('#canvas-scroll');
    if (graph) zoom = Math.max(overview ? 0.35 : 0.85, Math.min(1, (area.clientWidth - 40) / parseFloat(graph.style.width), overview ? (area.clientHeight - 40) / parseFloat(graph.style.height) : 1));
    renderCanvas(); area.scrollTo(0, 0); if (!overview) revealSelected();
  }
  $('#zoom-in').addEventListener('click', () => changeZoom(0.1)); $('#zoom-out').addEventListener('click', () => changeZoom(-0.1));
  $('#zoom-fit').addEventListener('click', () => fitCanvas());
  let resizeFrame = 0;
  window.addEventListener('resize', () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => { if (view === 'builder') renderCanvas(); }); });
  let pan = null;
  const canvasArea = $('#canvas-scroll');
  canvasArea.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0 || event.target.closest('button')) return;
    const bounds = canvasArea.getBoundingClientRect();
    if (event.clientX > bounds.right - 16 || event.clientY > bounds.bottom - 16) return;
    canvasArea.focus({preventScroll: true});
    pan = {x: event.clientX, y: event.clientY, left: canvasArea.scrollLeft, top: canvasArea.scrollTop};
    canvasArea.setPointerCapture(event.pointerId); canvasArea.classList.add('is-panning'); event.preventDefault();
  });
  canvasArea.addEventListener('pointermove', event => { if (pan) canvasArea.scrollTo(pan.left - event.clientX + pan.x, pan.top - event.clientY + pan.y); });
  const endPan = () => { pan = null; canvasArea.classList.remove('is-panning'); };
  canvasArea.addEventListener('pointerup', endPan); canvasArea.addEventListener('pointercancel', endPan); canvasArea.addEventListener('lostpointercapture', endPan);
  document.addEventListener('keydown', event => {
    if (!(event.ctrlKey || event.metaKey)) return;
    if (event.key.toLowerCase() === 's') { event.preventDefault(); if (hasDraft()) { toast('Apply or discard the editor draft before exporting.'); openDraft(); } else if ($('#export').disabled) toast('Resolve configuration errors before exporting.'); else $('#export').click(); }
    if (event.key.toLowerCase() === 'z' && !event.target.matches('input,textarea,select')) { event.preventDefault(); undo(event.shiftKey); }
  });
  $('#edit-selected').addEventListener('click', () => { $('.inspector').scrollIntoView({block: 'start'}); $('#node-id').focus({preventScroll: true}); });
  window.addEventListener('beforeunload', event => { if (!storageAvailable && (history.length || hasDraft())) { event.preventDefault(); event.returnValue = ''; } });
  $('#catalog-summary').textContent = `${catalog.training.length} training · ${catalog.evaluation.length} evaluation profiles`;
  $('#catalog-info').addEventListener('click', () => {
    $('#catalog-detail').innerHTML = `<p>Built from resolved, validated repository YAMLs. ${catalog.training.length} training profiles, ${catalog.evaluation.length} evaluation profiles and ${Object.values(catalog.components).reduce((count, items) => count + items.length, 0)} reusable components.</p><p><strong>Supported:</strong> hierarchical groups, registered plugin training and evaluation, declared quantization choices, task selection, inherited patches, checkpoint handoff, dependencies, YAML/JSON workflow import, YAML export and local undo/redo.</p><p>Model capabilities and checkpoint compatibility come from the registered plugin manifests.</p><p>Browser checks do not replace the full CLI schema validation and cannot inspect your GPU, dataset or model files.</p><div class="command-box">Refresh after repository configuration changes:<br><code>python deploy/scripts/build-workflow-catalog.py</code></div><p class="muted">Generated: ${esc(catalog.generated_at)}<br>Source identity: ${esc(catalog.source_sha256.slice(0, 16))}</p>${catalog.excluded.length ? '<p>Excluded unsupported examples:</p><ul>' + catalog.excluded.map(item => '<li>' + esc(item.ref) + '</li>').join('') + '</ul>' : ''}`;
    $('#info-dialog').showModal();
  });
  window.OVLABStudio = {getDocument: () => C.clone(workflow), getAnalysis: () => C.analyze(workflow, catalog), load: document => replaceWorkflow(document, 'Workflow loaded.')};
  render(); persist(); requestAnimationFrame(() => fitCanvas(false));
})();
