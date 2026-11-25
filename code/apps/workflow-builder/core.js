/* Pure workflow operations shared by the offline editor and its regressions. */
(function (global) {
  'use strict';
  const PHASES = ['train', 'evaluate'];
  const RESERVED = new Set(['train', 'evaluate', 'derived', 'interactive', 'isolated', 'grouped']);
  const SAFE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const clone = value => JSON.parse(JSON.stringify(value));
  function guard(value, depth = 0) {
    if (depth > 100) throw new Error('Document nesting exceeds the supported limit.');
    if (value && typeof value === 'object') for (const key of Object.keys(value)) {
      if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error(`Unsupported key: ${key}`);
      guard(value[key], depth + 1);
    }
    return value;
  }
  function merge(parent, patch) {
    guard(patch);
    if (!object(parent) || !object(patch)) return clone(patch);
    const result = clone(parent);
    for (const [key, value] of Object.entries(patch)) result[key] = object(result[key]) && object(value) ? merge(result[key], value) : clone(value);
    return result;
  }
  function get(value, path) { return path.split('.').reduce((item, key) => item == null ? undefined : item[key], value); }
  function set(value, path, entry) {
    const keys = path.split('.');
    if (keys.some(key => ['__proto__', 'constructor', 'prototype'].includes(key))) throw new Error('Unsupported field.');
    if (entry === undefined) {
      function remove(item, index) {
        if (!object(item)) return;
        if (index === keys.length - 1) delete item[keys[index]];
        else {
          remove(item[keys[index]], index + 1);
          if (object(item[keys[index]]) && !Object.keys(item[keys[index]]).length) delete item[keys[index]];
        }
      }
      remove(value, 0);
    } else {
      let cursor = value;
      for (const key of keys.slice(0, -1)) { if (!object(cursor[key])) cursor[key] = {}; cursor = cursor[key]; }
      cursor[keys.at(-1)] = entry;
    }
  }
  function flatten(root) {
    const result = [];
    function visit(node, path, parent, depth, inherited) {
      const state = clone(inherited || {});
      for (const phase of PHASES) if (node.config && node.config[phase]) {
        const local = node.config[phase], next = state[phase] || {patch: {}, component_patches: {}};
        if (own(local, 'profile')) { next.profile = local.profile; next.profileSource = path; }
        if (own(local, 'checkpoint_from')) { next.checkpoint_from = local.checkpoint_from; next.checkpointSource = path; }
        next.patch = merge(next.patch || {}, local.patch || {});
        next.component_patches = merge(next.component_patches || {}, local.component_patches || {});
        state[phase] = next;
      }
      result.push({node, path, parent, depth, state});
      for (const child of node.children || []) visit(child, path ? `${path}/${child.id}` : child.id, path, depth + 1, state);
    }
    visit(root, '', null, 0, {});
    return result;
  }
  function profile(catalog, phase, reference) {
    const prefix = phase === 'train' ? 'experiments/training/' : 'experiments/evaluation/';
    return (catalog[phase === 'train' ? 'training' : 'evaluation'] || []).find(item => item.ref === reference || item.ref === prefix + reference);
  }
  function pluginForPolicy(catalog, policy) {
    if (!policy) return undefined;
    if (policy.provider) return catalog.plugins?.[policy.provider];
    const matches = Object.values(catalog.plugins || {}).filter(item => item.config_type === policy.type);
    return matches.length === 1 ? matches[0] : undefined;
  }
  function resolve(entry, phase, catalog) {
    const state = entry.state[phase];
    if (!state || !state.profile) return null;
    const source = profile(catalog, phase, state.profile);
    if (!source) return {unavailable: true, reference: state.profile};
    const document = merge(source.document, state.patch || {}), components = {};
    const invalid = message => ({unavailable: true, reference: state.profile, error: message});
    if (document.kind !== (phase === 'train' ? 'training' : 'experiment') || document.schema_version !== (phase === 'train' ? 'ovlab.training-profile/v1' : '0.1.0')) return invalid('Patches must preserve the profile kind and schema version.');
    if (phase === 'train' && (!object(document.training) || !object(document.model) || !object(document.dataset) || typeof document.dataset.ref !== 'string' || (document.training.mode === 'peft' && !object(document.training.peft)))) return invalid('Training patches must preserve the training, model and dataset objects.');
    const trainingPlugin = phase === 'train' && Object.values(catalog.plugins || {}).find(item => item.project === document.trainer?.id && (item.capabilities?.training?.implementations || []).includes(document.trainer?.implementation));
    const method = document.training?.mode === 'peft' ? document.training.peft?.method : 'full';
    if (phase === 'train' && (!['full', 'peft'].includes(document.training.mode) || typeof method !== 'string' || (trainingPlugin && !(trainingPlugin.capabilities.training.methods || []).includes(method)))) return invalid('Choose a supported method declared by the training plugin.');
    if (phase === 'evaluate' && (!object(document.components) || !object(document.experiment) || ['policy', 'benchmark', 'protocol', 'metrics', 'artifacts'].some(kind => typeof document.components[kind] !== 'string'))) return invalid('Evaluation defaults need experiment details and explicit policy, benchmark, protocol, metrics and artifacts components.');
    if (phase === 'evaluate') for (const [kind, reference] of Object.entries(document.components || {})) {
      const component = (catalog.components[kind] || []).find(item => item.ref === reference);
      if (component) components[kind] = merge(component.document, (state.component_patches || {})[kind] || {});
    }
    if (phase === 'evaluate') {
      for (const kind of ['policy', 'benchmark']) if (components[kind] && (!object(components[kind]) || !object(components[kind].settings))) return invalid(`${kind} patches must preserve their settings object.`);
      if (components.policy && components.policy.type !== 'mock' && !pluginForPolicy(catalog, components.policy)) return invalid('Unknown policy type; choose a registered policy component.');
      if (components.benchmark && !(catalog.components.benchmark || []).some(item => item.document.type === components.benchmark.type)) return invalid('Unknown benchmark type; choose a catalog benchmark component.');
      if (components.protocol && (!object(components.protocol) || !object(components.protocol.execution))) return invalid('Protocol patches must preserve execution settings.');
    }
    return {document, components, source, state, plugin: phase === 'evaluate' ? pluginForPolicy(catalog, components.policy) : trainingPlugin};
  }
  function family(resolved, phase) {
    if (!resolved || resolved.unavailable) return '';
    if (phase === 'train') return resolved.document.training.mode === 'peft' ? resolved.document.training.peft.method : 'full';
    return resolved.plugin?.family || resolved.components.policy?.type || 'mock';
  }
  function bindingCompatible(trained, evaluated) {
    if (!trained || trained.unavailable || !evaluated || evaluated.unavailable) return false;
    const binding = evaluated.plugin?.capabilities?.checkpoint_binding;
    return Boolean(binding && (binding.training_methods || []).includes(family(trained, 'train')) &&
      (!evaluated.plugin.project || evaluated.plugin.project === trained.document.trainer?.id));
  }
  function sameDocument(a, b) {
    function canonical(item) { return Array.isArray(item) ? item.map(canonical) : object(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, canonical(item[key])])) : item; }
    return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
  }
  function analyze(root, catalog) {
    const issues = [], rows = [], entries = [], tasks = new Map();
    let nodeCount = 0;
    const issue = (severity, path, message, extra = {}) => issues.push({severity, path, message, ...extra});
    if (!object(root)) return {issues: [{severity: 'error', path: '', message: 'The workflow must be an object.'}], entries, rows};
    const allowedNode = ['id', 'name', 'config', 'execute', 'depends_on', 'children'];
    function inspect(node, path, depth) {
      if (++nodeCount > 512) { if (nodeCount === 513) issue('error', path, 'Workflows support at most 512 nodes.'); return; }
      if (!object(node)) { issue('error', path, 'Every node must be an object.'); return; }
      if (typeof node.id !== 'string' || !SAFE.test(node.id) || RESERVED.has(node.id)) issue('error', path, 'Use a portable node ID (letters, digits, dots, _ and -), excluding reserved phase/directory names.');
      if (node.name !== undefined && (typeof node.name !== 'string' || !node.name.trim())) issue('error', path, 'Node name must be a nonempty string.');
      for (const key of Object.keys(node)) if (!(depth ? allowedNode : [...allowedNode, 'schema_version', 'kind', 'execution']).includes(key)) issue('error', path, `Unknown node field: ${key}`);
      if (depth > 12) { issue('error', path, 'Workflows support at most 12 levels.'); return; }
      if (node.execute !== undefined && (!Array.isArray(node.execute) || node.execute.some(p => !PHASES.includes(p)) || new Set(node.execute).size !== node.execute.length)) issue('error', path, 'Execute must contain unique train/evaluate phases.');
      if (node.config !== undefined) {
        if (!object(node.config)) issue('error', path, 'Configuration must be an object.');
        else for (const [phase, value] of Object.entries(node.config)) {
          if (!PHASES.includes(phase) || !object(value)) { issue('error', path, `Invalid phase configuration: ${phase}`); continue; }
          for (const key of Object.keys(value)) if (!(phase === 'train' ? ['profile', 'patch'] : ['profile', 'patch', 'component_patches', 'checkpoint_from']).includes(key)) issue('error', path, `Unknown ${phase} configuration field: ${key}`);
          if (own(value, 'profile') && (typeof value.profile !== 'string' || !value.profile)) issue('error', path, 'Profile references cannot be empty.');
          for (const key of ['patch', 'component_patches']) if (own(value, key) && !object(value[key])) issue('error', path, `${phase}.${key} must be an object.`);
          if (value.component_patches && Object.keys(value.component_patches).some(key => !['benchmark', 'policy', 'protocol', 'metrics', 'artifacts'].includes(key) || !object(value.component_patches[key]))) issue('error', path, 'Component patches must be objects for supported component kinds.');
          if (own(value, 'checkpoint_from') && (typeof value.checkpoint_from !== 'string' || (value.checkpoint_from !== '' && value.checkpoint_from.split('/').some(part => !SAFE.test(part))))) issue('error', path, 'Checkpoint source must be a root-relative node path.');
        }
      }
      if (node.depends_on !== undefined && (!Array.isArray(node.depends_on) || node.depends_on.some(dep => !object(dep) || Object.keys(dep).some(key => !['node', 'phase'].includes(key)) || typeof dep.node !== 'string' || !PHASES.includes(dep.phase)))) issue('error', path, 'Dependencies require a node path and a train/evaluate phase.');
      if (node.children !== undefined && !Array.isArray(node.children)) issue('error', path, 'Children must be a list.');
      const names = new Set();
      for (const child of Array.isArray(node.children) ? node.children : []) {
        if (names.has(child?.id)) issue('error', path, `Duplicate child ID: ${child?.id}`);
        names.add(child?.id);
        inspect(child, path ? `${path}/${child?.id}` : String(child?.id), depth + 1);
      }
    }
    if (root.schema_version !== 'ovlab.experiment/v1' || root.kind !== 'workflow') issue('error', '', 'Use an ovlab.experiment/v1 workflow document.');
    if (typeof root.name !== 'string' || !root.name.trim()) issue('error', '', 'Set a workflow name.');
    if (root.execution !== undefined && (!object(root.execution) || Object.keys(root.execution).some(k => k !== 'max_workers') || (root.execution.max_workers !== undefined && (!Number.isInteger(root.execution.max_workers) || root.execution.max_workers < 1 || root.execution.max_workers > 32)))) issue('error', '', 'Worker count must be an integer between 1 and 32.');
    inspect(root, '', 0);
    if (issues.some(item => item.severity === 'error')) return {issues, entries, rows};
    entries.push(...flatten(root));
    if (entries.length > 512) issue('error', '', 'Workflows support at most 512 nodes.');
    const byPath = new Map(entries.map(entry => [entry.path, entry]));
    for (const entry of entries) for (const phase of PHASES) {
      const resolved = resolve(entry, phase, catalog), executes = (entry.node.execute || []).includes(phase);
      if (executes && !resolved) issue('error', entry.path, `${phase} has no inherited or local profile.`);
      if (resolved?.unavailable) issue('error', entry.path, resolved.error || `Profile is outside this catalog: ${resolved.reference}. Refresh the catalog or select a listed profile.`);
      if (resolved && !resolved.unavailable) {
        const doc = resolved.document;
        if (phase === 'train') {
          for (const key of ['max_steps', 'per_device_batch_size', 'gradient_accumulation_steps', 'validation_batches']) if (doc.training[key] !== undefined && (!Number.isInteger(doc.training[key]) || doc.training[key] < 1)) issue('error', entry.path, `${key} must be a positive integer.`);
          if (!(Number.isFinite(doc.training.learning_rate) && doc.training.learning_rate > 0)) issue('error', entry.path, 'Learning rate must be positive.');
          if (!['bf16', 'fp32'].includes(doc.training.precision)) issue('error', entry.path, 'Training precision must be bf16 or fp32.');
          const groups = doc.dataset.partition?.groups, suite = (doc.dataset.ref || '').split('/').at(-1);
          if (groups && (!Array.isArray(groups) || !groups.length || groups.some(group => typeof group !== 'string' || (catalog.tasks[suite]?.length && !catalog.tasks[suite].includes(group))))) issue('error', entry.path, 'Training task groups must be nonempty and belong to the dataset suite.');
        } else {
          for (const [kind, reference] of Object.entries(doc.components || {})) if (!resolved.components[kind]) issue('error', entry.path, `Unknown ${kind} component: ${reference}`);
          const policy = resolved.components.policy, benchmark = resolved.components.benchmark, protocol = resolved.components.protocol;
          const descriptor = resolved.plugin, quantization = descriptor?.deployment?.quantization;
          if (quantization && !quantization.includes(policy?.settings?.runtime?.quantization)) issue('error', entry.path, `Unsupported inference quantization. This plugin supports: ${quantization.join(', ')}.`);
          const expectedDeployment = descriptor?.deployment?.profile;
          if (expectedDeployment && doc.deployment && doc.deployment.profile !== expectedDeployment) issue('error', entry.path, `This policy requires deployment.profile: ${expectedDeployment}.`);
          if (doc.deployment && !['egl', 'glfw'].includes(doc.deployment.renderer)) issue('error', entry.path, 'Renderer must be egl or glfw.');
          for (const key of ['rollouts_per_task', 'maximum_episode_steps']) if (protocol && (!Number.isInteger(protocol.execution[key]) || protocol.execution[key] < 1)) issue('error', entry.path, `${key} must be a positive integer.`);
          const indices = benchmark?.settings?.task_indices;
          if (indices !== undefined && indices !== 'all' && (!Array.isArray(indices) || !indices.length || new Set(indices).size !== indices.length || indices.some(n => !Number.isInteger(n) || n < 0 || (catalog.tasks[benchmark.settings.suite] && n >= catalog.tasks[benchmark.settings.suite].length)))) issue('error', entry.path, 'Task indices must be unique, nonnegative integers within the selected suite.');
          const identity = descriptor?.capabilities?.checkpoint_binding?.adapter_identity;
          if (executes && identity && !own(entry.state.evaluate, 'checkpoint_from')) {
            const adapter = get(policy.settings, identity.settings_path);
            if (!adapter || !adapter[identity.path] || identity.hashes.some(key => !/^[a-f0-9]{64}$/.test(adapter[key]) || /^0{64}$/.test(adapter[key]))) issue('error', entry.path, 'This policy needs a training checkpoint source or an existing adapter path with real SHA256 hashes.');
            else issue('info', entry.path, 'Existing adapter selected. Its files and hashes are checked at runtime.');
          }
        }
      }
      if (executes) tasks.set(JSON.stringify([entry.path, phase]), {entry, phase, resolved, dependencies: new Set()});
    }
    for (const [key, task] of tasks) {
      const {entry, phase, resolved} = task;
      const depend = (path, otherPhase) => {
        const other = JSON.stringify([path, otherPhase]);
        if (!tasks.has(other)) issue('error', entry.path, `Dependency ${path || '<root>'}:${otherPhase} does not execute.`);
        else task.dependencies.add(other);
      };
      if (phase === 'evaluate' && tasks.has(JSON.stringify([entry.path, 'train']))) depend(entry.path, 'train');
      for (const dependency of entry.node.depends_on || []) depend(dependency.node, dependency.phase);
      if (phase === 'evaluate' && own(entry.state.evaluate, 'checkpoint_from')) {
        const sourcePath = entry.state.evaluate.checkpoint_from;
        depend(sourcePath, 'train');
        const source = byPath.get(sourcePath), trained = source && resolve(source, 'train', catalog);
        if (trained && !trained.unavailable && resolved && !resolved.unavailable) {
          const method = family(trained, 'train'), binding = resolved.plugin?.capabilities?.checkpoint_binding;
          if (!bindingCompatible(trained, resolved)) issue('error', entry.path, `Checkpoint source method ${method} is incompatible with this evaluation policy.`);
          if (!resolved.document.deployment) issue('error', entry.path, 'Checkpoint handoff requires a deployed evaluation policy.');
          if (binding?.normalization_key && resolved.components.policy?.settings.unnorm_key !== binding.normalization_key) issue('error', entry.path, `This checkpoint binding requires normalization: ${binding.normalization_key}.`);
          if (binding?.match_training_normalization && trained.document.training.normalization?.unnorm_key !== resolved.components.policy?.settings.unnorm_key) issue('error', entry.path, 'Training and evaluation normalization keys must match.');
        }
      }
      if (task.dependencies.has(key)) issue('error', entry.path, 'A phase cannot depend on itself.');
      if (phase === 'evaluate' && resolved && !resolved.unavailable) {
        for (const child of entries.filter(other => other.path.startsWith(entry.path ? entry.path + '/' : '') && other.path !== entry.path)) {
          if (!(child.node.execute || []).includes(phase)) continue;
          const childResolved = resolve(child, phase, catalog);
          const signature = (configuration, state) => ({components: configuration.components, resources: configuration.document.resources, deployment: configuration.document.deployment, checkpoint: state.checkpoint_from});
          if (childResolved && !childResolved.unavailable && sameDocument(signature(resolved, entry.state.evaluate), signature(childResolved, child.state.evaluate))) issue('warning', entry.path, `This node and ${child.path} execute the same evaluation. Remove this node's Evaluate phase if it is only a group.`, {fix: 'remove-evaluate', related: child.path});
        }
      }
    }
    const pending = new Map([...tasks].map(([key, value]) => [key, new Set(value.dependencies)]));
    while (pending.size) {
      const ready = [...pending].find(([, deps]) => !deps.size);
      if (!ready) { issue('error', '', 'Phase dependencies contain a cycle.'); break; }
      const [key] = ready, task = tasks.get(key);
      rows.push({...task, key, path: task.entry.path, checkpoint: task.phase === 'evaluate' ? task.entry.state.evaluate?.checkpoint_from : undefined});
      pending.delete(key);
      for (const dependencies of pending.values()) dependencies.delete(key);
    }
    if (!tasks.size) issue('warning', '', 'No phases execute yet. Add a training or evaluation node.');
    if ((root.execution?.max_workers || 2) > 1 && tasks.size) issue('info', '', 'Parallel workers do not reserve GPUs or VRAM. Match the worker count to your hardware.');
    const scheduled = new Set(rows.map(row => row.key));
    const blockedRows = [...tasks].filter(([key]) => !scheduled.has(key)).map(([key, task]) => ({...task, key, path: task.entry.path, checkpoint: task.phase === 'evaluate' ? task.entry.state.evaluate?.checkpoint_from : undefined, blocked: true}));
    return {issues, rows, blockedRows, entries, phaseCount: tasks.size};
  }
  function rewriteReferences(root, oldPath, newPath) {
    const rewrite = path => path === oldPath ? newPath : oldPath && path.startsWith(oldPath + '/') ? newPath + path.slice(oldPath.length) : path;
    for (const {node} of flatten(root)) {
      if (own(node.config?.evaluate, 'checkpoint_from')) node.config.evaluate.checkpoint_from = rewrite(node.config.evaluate.checkpoint_from);
      for (const dependency of node.depends_on || []) dependency.node = rewrite(dependency.node);
    }
  }
  function uniqueId(parent, prefix = 'node') {
    prefix = prefix.slice(0, 110);
    const taken = new Set((parent.children || []).map(child => child.id));
    if (!taken.has(prefix) && !RESERVED.has(prefix)) return prefix;
    let index = 2; while (taken.has(`${prefix}-${index}`)) index++;
    return `${prefix}-${index}`;
  }
  function moveNode(root, path, parentPath) {
    if (!path || path === parentPath || parentPath.startsWith(path + '/')) throw new Error('A node cannot be moved inside itself or its descendants.');
    const entries = flatten(root), source = entries.find(entry => entry.path === path), parent = entries.find(entry => entry.path === parentPath);
    if (!source || !parent) throw new Error('Node or destination is unavailable.');
    if ((parent.node.children || []).some(child => child !== source.node && child.id === source.node.id)) throw new Error('The destination already has a child with this ID.');
    const deepest = Math.max(...entries.filter(entry => entry.path === path || entry.path.startsWith(path + '/')).map(entry => entry.depth - source.depth));
    if (parent.depth + 1 + deepest > 12) throw new Error('This move would exceed 12 hierarchy levels.');
    const previous = entries.find(entry => entry.path === source.parent).node;
    previous.children.splice(previous.children.indexOf(source.node), 1);
    (parent.node.children ||= []).push(source.node);
    const next = parentPath ? `${parentPath}/${source.node.id}` : source.node.id;
    rewriteReferences(root, path, next);
    return next;
  }
  function renameNode(root, path, id) {
    if (!SAFE.test(id) || RESERVED.has(id)) throw new Error('Choose a portable ID that is not a reserved directory or phase name.');
    const entries = flatten(root), entry = entries.find(item => item.path === path);
    if (entry.parent !== null && entries.find(item => item.path === entry.parent).node.children.some(child => child !== entry.node && child.id === id)) throw new Error('A sibling already uses this ID.');
    entry.node.id = id;
    const next = path ? (entry.parent ? entry.parent + '/' : '') + id : '';
    if (path) rewriteReferences(root, path, next);
    return next;
  }
  function dumpYaml(value) {
    guard(value);
    const lines = [];
    const scalar = item => item === null ? 'null' : typeof item === 'string' ? JSON.stringify(item) : String(item);
    function emit(item, indent) {
      const prefix = ' '.repeat(indent);
      if (object(item)) for (const [key, child] of Object.entries(item)) {
        if (!/^[A-Za-z0-9_./-]+$/.test(key) || key.startsWith('/') || key.endsWith('/') || key.split('/').includes('..') || key.includes('//')) throw new Error(`Unsupported YAML key: ${key}`);
        if (object(child) && !Object.keys(child).length) lines.push(`${prefix}${key}: {}`);
        else if (Array.isArray(child) && (!child.length || child.every(value => !object(value) && !Array.isArray(value)))) lines.push(`${prefix}${key}: [${child.map(scalar).join(', ')}]`);
        else if (child && typeof child === 'object') { lines.push(`${prefix}${key}:`); emit(child, indent + 2); }
        else lines.push(`${prefix}${key}: ${scalar(child)}`);
      } else if (Array.isArray(item)) for (const child of item) {
        if (object(child)) {
          const start = lines.length; emit(child, indent + 2);
          if (lines.length === start) throw new Error('Empty mappings in sequences are unsupported.');
          lines[start] = prefix + '- ' + lines[start].slice(indent + 2);
        } else if (Array.isArray(child)) throw new Error('Nested YAML sequences are unsupported.');
        else lines.push(`${prefix}- ${scalar(child)}`);
      }
    }
    emit(value, 0); return lines.join('\n') + '\n';
  }
  function parse(text) {
    if (text.length > 2000000) throw new Error('Configurations must be smaller than 2 MB.');
    if (text.trimStart().startsWith('{')) {
      const parsed = guard(JSON.parse(text));
      // JSON.parse silently overwrites duplicate keys. The CLI rejects them,
      // so inspect the already validated JSON token stream before accepting it.
      let cursor = 0;
      const whitespace = () => { while (/\s/.test(text[cursor] || '') && cursor < text.length) cursor++; };
      function stringToken() {
        const start = cursor++;
        while (cursor < text.length) {
          if (text[cursor] === '\\') cursor += 2;
          else if (text[cursor++] === '"') return JSON.parse(text.slice(start, cursor));
        }
      }
      function visit() {
        whitespace();
        if (text[cursor] === '{') {
          cursor++; whitespace(); const names = new Set();
          while (text[cursor] !== '}') {
            const key = stringToken();
            if (names.has(key)) throw new Error(`Duplicate JSON key: ${key}`);
            names.add(key); whitespace(); cursor++; visit(); whitespace();
            if (text[cursor] !== ',') break;
            cursor++; whitespace();
          }
          cursor++;
        } else if (text[cursor] === '[') {
          cursor++; whitespace();
          while (text[cursor] !== ']') { visit(); whitespace(); if (text[cursor] !== ',') break; cursor++; }
          cursor++;
        } else if (text[cursor] === '"') stringToken();
        else while (cursor < text.length && !/[\s,\]}]/.test(text[cursor])) cursor++;
      }
      visit(); return parsed;
    }
    const lines = [];
    function fail(number, message) { throw new Error(`Line ${number}: ${message}`); }
    function scan(value, callback) {
      let quote = '', escaped = false, depth = 0;
      for (let index = 0; index < value.length; index++) {
        const char = value[index];
        if (escaped) { escaped = false; continue; }
        if (quote === '"' && char === '\\') { escaped = true; continue; }
        if (char === '"' || char === "'") { if (!quote) quote = char; else if (quote === char) quote = ''; continue; }
        if (!quote) { if (callback(char, index, depth)) return index; if (char === '[') depth++; else if (char === ']') depth--; }
      }
      return quote ? -2 : -1;
    }
    text.split(/\r?\n/).forEach((raw, index) => {
      const number = index + 1;
      if (raw.includes('\t')) fail(number, 'Tabs are not supported.');
      const comment = scan(raw, (char, i) => char === '#' && (!i || /\s/.test(raw[i-1])));
      if (comment === -2) fail(number, 'Unterminated quote.');
      raw = (comment >= 0 ? raw.slice(0, comment) : raw).trimEnd();
      if (!raw.trim()) return;
      const indent = raw.length - raw.trimStart().length, content = raw.trimStart();
      if (indent % 2 || indent > 200) fail(number, 'Use two-space indentation.');
      if (/^(---|\.\.\.|%YAML|%TAG)/.test(content)) fail(number, 'YAML document markers and directives are unsupported.');
      lines.push({number, indent, content});
    });
    if (!lines.length || lines[0].indent) throw new Error('Start the configuration at column zero.');
    function scalar(raw, number) {
      if (raw.startsWith('[')) {
        if (!raw.endsWith(']')) fail(number, 'Unterminated inline list.');
        const inner = raw.slice(1, -1); if (!inner.trim()) return [];
        const parts = []; let start = 0;
        scan(inner, (char, index) => { if (char === ',') { parts.push(inner.slice(start, index).trim()); start = index + 1; } return false; });
        parts.push(inner.slice(start).trim());
        if (parts.some(part => !part)) fail(number, 'Empty inline list item.');
        return parts.map(part => scalar(part, number));
      }
      if (raw === '{}') return {};
      if (raw.startsWith('{') || raw.endsWith('}')) fail(number, 'Inline mappings are unsupported.');
      if (raw.startsWith('"')) { const value = JSON.parse(raw); if (typeof value !== 'string') fail(number, 'Expected a quoted string.'); return value; }
      if (raw.startsWith("'")) { if (!raw.endsWith("'")) fail(number, 'Unterminated single quote.'); return raw.slice(1, -1).replace(/''/g, "'"); }
      if (/^(null|~)$/i.test(raw)) return null;
      if (/^(true|false)$/i.test(raw)) return raw.toLowerCase() === 'true';
      if (/^[-+]?(0|[1-9][0-9]*)$/.test(raw) || /^[-+]?(?:\d+\.\d*|\d*\.\d+|\d+[eE][-+]?\d+|\d+\.\d*[eE][-+]?\d+)$/.test(raw)) { const value = Number(raw); if (!Number.isFinite(value)) fail(number, 'Numbers must be finite.'); return value; }
      if (/^[&*!|>]|^<<|(?:^|\s)[&*!]\S+/.test(raw)) fail(number, 'YAML anchors, aliases, tags, merges and block strings are unsupported.');
      return raw;
    }
    function mapping(text, number) {
      const colon = scan(text, (char, index, depth) => char === ':' && depth === 0);
      if (colon < 0) return null;
      const key = text.slice(0, colon).trim(), raw = text.slice(colon + 1).trim();
      if (!/^[A-Za-z0-9_./-]+$/.test(key) || key.startsWith('/') || key.endsWith('/') || key.split('/').includes('..') || key.includes('//') || ['__proto__', 'prototype', 'constructor'].includes(key)) fail(number, 'Unsupported mapping key.');
      return [key, raw];
    }
    function block(index, indent) {
      const sequence = lines[index].content.startsWith('- '), result = sequence ? [] : {};
      while (index < lines.length) {
        const line = lines[index];
        if (line.indent < indent) break;
        if (line.indent > indent) fail(line.number, 'Unexpected indentation.');
        if (sequence !== line.content.startsWith('- ')) break;
        const item = sequence ? line.content.slice(2).trim() : line.content;
        const pair = mapping(item, line.number);
        if (!pair) { if (!sequence) fail(line.number, 'Expected a mapping entry.'); result.push(scalar(item, line.number)); index++; continue; }
        const [key, raw] = pair; let value; index++;
        if (raw) value = scalar(raw, line.number);
        else {
          const nextIndent = indent + (sequence ? 4 : 2);
          if (!lines[index] || lines[index].indent !== nextIndent) fail(line.number, 'Key requires an indented value.');
          [value, index] = block(index, nextIndent);
        }
        if (sequence) {
          let entry = {[key]: value};
          if (lines[index]?.indent === indent + 2) {
            let continuation; [continuation, index] = block(index, indent + 2);
            if (!object(continuation)) fail(line.number, 'List mapping continuation must be an object.');
            if (own(continuation, key)) fail(line.number, `Duplicate key: ${key}`);
            entry = {...entry, ...continuation};
          }
          result.push(entry);
        } else { if (own(result, key)) fail(line.number, `Duplicate key: ${key}`); result[key] = value; }
      }
      return [result, index];
    }
    const [document, end] = block(0, 0);
    if (end !== lines.length) fail(lines[end].number, 'Unexpected indentation.');
    if (!object(document)) throw new Error('The root must be a mapping.');
    return guard(document);
  }
  const api = {PHASES, RESERVED, SAFE, own, object, clone, merge, get, set, flatten, profile, pluginForPolicy, resolve, family, bindingCompatible, sameDocument,
    analyze, rewriteReferences, uniqueId, moveNode, renameNode, dumpYaml, parse, guard};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.OVLABBuilder = api;
})(typeof window === 'undefined' ? globalThis : window);
