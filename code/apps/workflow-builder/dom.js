/* Keep form controls attached while their surrounding state is refreshed. */
(function (global) {
  'use strict';
  function key(node) {
    if (node.nodeType !== 1) return null;
    if (node.id) return 'id:' + node.id;
    if (node.dataset.fix) return 'fix:' + node.dataset.fix + ':' + node.dataset.fixPath;
    for (const name of ['section', 'inspectorKey', 'action', 'reset', 'trainTask', 'select']) {
      if (node.dataset[name] !== undefined) return name + ':' + node.dataset[name];
    }
    if (node.classList.contains('field')) {
      const control = node.querySelector(':scope > input, :scope > select, :scope > textarea');
      if (control?.id) return 'field:' + control.id;
    }
    return null;
  }
  function compatible(a, b) {
    return a.nodeType === b.nodeType && (a.nodeType !== 1 || (a.tagName === b.tagName && key(a) === key(b)));
  }
  function update(current, next) {
    if (current.nodeType === 3) { if (current.nodeValue !== next.nodeValue) current.nodeValue = next.nodeValue; return; }
    if (current.nodeType !== 1) return;
    for (const attribute of [...current.attributes]) if (!next.hasAttribute(attribute.name)) current.removeAttribute(attribute.name);
    for (const attribute of next.attributes) if (current.getAttribute(attribute.name) !== attribute.value) current.setAttribute(attribute.name, attribute.value);
    children(current, next);
    if (current instanceof HTMLInputElement) {
      if (current.value !== next.value) current.value = next.value;
      current.checked = next.checked;
    } else if (current instanceof HTMLSelectElement || current instanceof HTMLTextAreaElement) {
      if (current.value !== next.value) current.value = next.value;
    }
  }
  function children(current, next) {
    const existing = [...current.childNodes];
    const keyed = new Map(existing.filter(node => key(node)).map(node => [key(node), node]));
    const kept = new Set();
    let position = current.firstChild;
    for (const template of [...next.childNodes]) {
      let matched = key(template) ? keyed.get(key(template)) : position;
      if (matched && (!compatible(matched, template) || kept.has(matched))) matched = null;
      if (!matched) matched = template.cloneNode(true);
      if (matched !== position) current.insertBefore(matched, position);
      update(matched, template);
      kept.add(matched);
      position = matched.nextSibling;
    }
    for (const old of existing) if (!kept.has(old)) old.remove();
  }
  global.OVLABStudioDOM = {
    render(host, html) {
      const template = document.createElement('template');
      template.innerHTML = html;
      children(host, template.content);
    }
  };
})(window);
