# OVLAB Workflow Studio

A static visual editor for OVLAB experiment trees. Open it with:

```bash
ovlab editor
```

On WSL, the command opens the editor with the default Windows browser. It also
works from another directory. Alternatively, open [`index.html`](index.html)
directly in a modern browser. The checked-in HTML,
CSS, JavaScript and repository catalog work through `file://`; there is no
server, build step, CDN or network request.

## Build a workflow

1. Start from a repository workflow, import an existing workflow YAML/JSON, or
   choose **New**.
2. Select a node in the outline or canvas. **Add child node** creates a group,
   training phase, evaluation phase, or a train → evaluate branch. New phases
   reuse inherited settings. Without inherited evaluation settings, a new
   evaluation selects the nearest LoRA/QuIC training source automatically.
3. Select real training/evaluation profiles in the inspector. Override task
   groups, optimizer settings, evaluation tasks, episode limits, components,
   quantization and renderer as needed.
4. Set **Execute at this node** explicitly. A group with neither phase enabled
   only passes configuration to children. The editor warns when a parent and
   its descendant execute the same evaluation, even when their display names
   differ. **Evaluate only children** removes the parent's redundant evaluation
   while keeping its training phase.
5. Select **Checkpoint source** to bind evaluation to a training phase. The
   dependency is added automatically. Renaming or moving a node updates its
   checkpoint and explicit dependency references.
6. Review **Execution plan** and **YAML**, then **Export YAML**. Save the download
   under `experiments/workflows/` and validate it with the actual resolver:

   ```bash
   ovlab run experiments/workflows/my-experiment.yaml --plan
   ovlab run experiments/workflows/my-experiment.yaml
   ```

Move nodes by dragging them onto a parent in the outline or selecting a parent
in the inspector. Duplicate/delete act on the whole subtree. Undo/redo preserve
the document and selection. The browser stores the workflow and its unapplied
editor draft when local storage is available; YAML export is the portable saved
copy. **Advanced phase
configuration** edits the local `node.config` as JSON without losing fields
that do not have a dedicated control.

YAML and advanced JSON drafts survive node selection, tab changes and reloads.
The draft banner offers **Open draft** and **Discard edits**. Apply or discard
the draft before changing other settings, undoing or exporting; merely visiting
another node never drops its text. JSON parse/structure errors stay next to the
editor. Workflow replacement requires confirmation when a draft is present;
Undo restores that draft as well as the previous workflow.

The canvas opens at a readable scale. Drag its background to pan, use the zoom
buttons, or choose **Fit** for a complete overview. Zoom keeps the viewport's
center. Keyboard arrows navigate the outline. On smaller screens, **Edit node**
jumps directly to the selected node's inspector, and **New** stays available.

Execution previews include all configured phases, even when a dependency cycle
blocks scheduling. The table shows training settings, evaluation task/episode
coverage, inference precision and checkpoint sources. Errors block export;
warnings remain visible and offer direct fixes where applicable.

The inspector distinguishes local profile selection from inheritance. Empty
override fields use the inherited/profile value. **Reset** removes a local
override; an inherited override can still apply. Mappings merge recursively and
lists replace their inherited values, matching the CLI.

## Capabilities and validation

Profiles and components come from the repository, including resolved `extends`
chains. The catalog also contains LIBERO task names from its pinned source.
Supported UI options follow those profiles: LoRA/QuIC training, Vanilla/LoRA/OFT/
QuIC evaluation, native mock evaluation, and supported inference precision modes.
Checkpoint handoff supports LoRA and QuIC. QuIC inference supports full precision;
OFT training is unavailable.

Browser checks cover tree structure, inheritance, phase dependencies, cycles,
known checkpoint compatibility, quantization restrictions and common numerical
settings. They complement the Python schemas; **`ovlab run … --plan` remains the
authoritative validation**. The static editor cannot inspect GPU capacity, model
files, dataset availability or adapter hashes on disk. These are checked by the
CLI and runtime. Creating or exporting a workflow does not execute workloads.

Only workflow documents (`kind: workflow`, `ovlab.experiment/v1`) can be imported.
Select standalone training/evaluation YAMLs as profiles. Unknown profile paths
must first be added to the repository and included in a refreshed catalog. YAML
uses the same bounded subset as OVLAB: two-space indentation, mappings, sequences,
inline scalar lists and quoted/plain scalars. Comments/formatting are normalized
on export. Duplicate keys and unsupported YAML features are rejected.

## Refresh the repository catalog

After changing profiles, components or workflow examples, run from the repository
root:

```bash
python deploy/scripts/build-workflow-catalog.py
```

The generator uses the real Python configuration loaders, includes only supported
documents and writes `catalog.js`. It records the source identity and lists
excluded examples in **Catalog & supported features**. The browser reads a
snapshot; it does not automatically observe filesystem changes. No model or GPU
package is initialized during catalog generation.

## Checks

With Node.js installed:

```bash
node code/apps/workflow-builder/core.test.cjs
```

Browser and resolver parity tests live in
[`code/tests/workflow_builder/`](../../tests/workflow_builder/). Install pytest,
Playwright and its Chromium browser in a development environment, then run:

```bash
python -m playwright install chromium
bash deploy/scripts/test.sh code/tests/workflow_builder/
```

`OVLAB_NODE` can select a Node executable. Browser tests use only local files and
temporary downloads, and pass those downloads to the real workflow planner.

## Files

| File | Purpose |
| --- | --- |
| `index.html`, `styles.css` | Accessible, responsive editor shell and styling. |
| `app.js` | Node editing, canvas, forms, source editor and local draft history. |
| `core.js` | Pure inheritance, dependency planning, validation and YAML import/export. |
| `dom.js` | Updates live form controls without detaching the target of a click or losing focus. |
| `catalog.js` | Generated repository profiles, components, tasks and examples. |
| `core.test.cjs` | Dependency-free core regressions. |
