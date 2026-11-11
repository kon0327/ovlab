# Experiment workflow

An `ovlab.experiment/v1` YAML or JSON document describes one experiment tree. Any
node can run training, evaluation, both, or neither. Nodes without `execute`
group their children. The tree can be as deep as needed within the validated
limit of 12 levels and 512 nodes.

The model comparison example is available as [YAML](../experiments/workflows/libero-model-tree.yaml)
and [JSON](../experiments/workflows/libero-model-tree.json). It
has model branches, a task branch and quantization ablations. It also runs
training at the LoRA model node. Its evaluation profile points to a configured
checkpoint independently of that training run. `depends_on` controls execution
order only; it does not select a newly trained checkpoint. To evaluate the
checkpoint from a training run, configure a matching policy/checkpoint artifact
explicitly before launching evaluation.

Two evaluation-only examples use the same 4-bit merged OpenVLA-LoRA profile:

```bash
ovlab run experiment run experiments/workflows/libero10-lora-4bit-smoke.yaml
ovlab run experiment run experiments/workflows/libero10-lora-4bit-eval.yaml
```

The smoke test runs task 0 once, limited to two environment steps. It checks
the execution and reporting pipeline; its truncated result is not a meaningful
measurement of task success. The evaluation uses all ten LIBERO-10 tasks,
three episodes per task and the profile's 520-step limit. Both load the
configured `openvla-7b-finetuned-libero-10` checkpoint and do not train a model.
The checkpoint and the images required by the deployment must already be
available. Add `--dry-run` to validate either workflow without running it.

## Plan and run

```bash
ovlab run experiment plan experiments/workflows/libero-model-tree.yaml --json
ovlab run experiment run experiments/workflows/libero-model-tree.yaml --dry-run
ovlab run experiment run experiments/workflows/libero-model-tree.yaml
ovlab run experiment run experiments/workflows/libero-model-tree.yaml --node lora/task-0
ovlab run experiment run experiments/workflows/libero-model-tree.yaml --node lora --node-only
ovlab run experiment run experiments/workflows/libero-model-tree.yaml --sequential
ovlab run experiment run experiments/workflows/libero-model-tree.yaml --max-workers 4
```

`--spec` remains an equivalent named form. A command accepts either the
positional target or its named alias, and rejects specifying both.

Independent phases run in parallel by default, with at most two workers.
`--sequential` runs one phase at a time; `--max-workers N` sets a limit from
1 to 32 (`--max-workers 1` is equivalent to `--sequential`). These options
also apply to `plan` and `run --dry-run`, which show the selected mode. A node's
evaluation waits for its training, and explicit `depends_on` phases must finish
before dependent phases start. On failure, no additional phases are started;
active subprocesses are interrupted and completed evidence remains recorded.

The worker limit does not assign GPUs or reserve VRAM. Concurrent GPU phases
can contend for the same device; use `--sequential` when they cannot fit
together. The training preflight estimate applies to each individual run. In
particular, the example's non-quantized LoRA profile estimates 27 GiB and is
rejected on a 24 GiB device even with `--sequential`. Select a compatible
training profile or a GPU with more memory. The QuIC BF16 profile has a 23 GiB
planning estimate, but represents a different adaptation method and does not
guarantee a particular peak for every configuration.

`--node` selects that node and its descendants. `--node-only` selects just
that node. A selected subtree must contain every phase it names in
`depends_on`; otherwise planning fails before any workload starts. `plan` and
`run --dry-run` validate the selected subtree and write nothing. The tree
shape is always checked, while unrelated siblings need not have runnable
profiles when a node is selected. Running creates
a new instance whose directory name contains the spec ID, UTC timestamp and
short content identity.

The plan contains each node's complete resolved train/evaluate configuration
and a source for every field. A canonical `spec.json` is saved under
`runs/<instance>/.experiment/`. When the input is YAML, `spec.yaml` also
preserves the original file, including comments. `result.json` records each phase's run ID,
canonical path, CLI result, comparison report and any failure. The export root
has an `experiment.json` index. Completed runs remain authoritative if a later
node or report fails.

## Workflow fields

| Field | Purpose |
| --- | --- |
| `schema_version` | Required at the root; `ovlab.experiment/v1`. |
| `kind` | Required at the root; `workflow`. |
| `id`, `name` | Portable ID and readable title. `id` is required at every node; `name` at the root. |
| `config.train.profile` | Existing YAML under `experiments/training/`. |
| `config.evaluate.profile` | Existing YAML under `experiments/evaluation/`. |
| `config.train.patch` | Changes fields in the resolved training profile, such as `dataset`, `model`, `training` or `resources`. |
| `config.evaluate.patch` | Changes fields in the resolved evaluation YAML, including `components` references and `deployment`. |
| `config.evaluate.component_patches` | Changes validated benchmark, policy, metrics, protocol or artifacts YAML; for example `benchmark.settings.task_indices`. |
| `execute` | Explicit array containing `train`, `evaluate` or both. This field is **not inherited**. |
| `depends_on` | Array of `{ "node": "root-relative/path", "phase": "train" }` scheduling dependencies. Use `""` for the root node. |
| `children` | Array of nested nodes; each child may override inherited configuration. |

Configuration objects inherit through the tree. Mappings merge recursively;
arrays and scalar values replace their parents. A child can select a new
`profile` and still receives inherited patches, so the resulting profile is
validated in full. Profile references and patched components must remain
inside `configs/`; arbitrary file paths, unknown fields, invalid component
combinations, missing dependencies and dependency cycles fail at planning.

Files ending in `.yaml` or `.yml` use the same strict YAML subset as other
OVLAB configurations: two-space indentation, mappings, sequences, inline
scalar lists, and quoted or plain scalars. Duplicate keys, anchors, aliases,
tags, merge keys, multi-document files and inline mappings are rejected. Other
extensions continue to be parsed as JSON. Both formats produce the same plan,
configuration inheritance and canonical spec hash for the same data.

Each phase uses the existing training or evaluation owner. For execution, the
workflow temporarily materializes the resolved YAML under the appropriate
`experiments/**/.generated/` or `configs/**/.generated/` directory, then removes it. The immutable plan and
canonical run evidence retain the exact resolved documents.

## Artifacts and comparisons

```text
runs/<instance>/
  .experiment/{spec,plan,result}.json
  .experiment/spec.yaml  # original input when a YAML file was used
  train/<node-path>/<run-id>/
  evaluate/<node-path>/<run-id>/

exports/<instance>/
  experiment.json
  train/<node-path>/{isolated,interactive,grouped}/
  evaluate/<node-path>/{isolated,interactive,grouped}/
```

Every completed node gets an isolated export and an interactive HTML report.
For each parent with results in at least two child branches, the workflow
generates grouped comparisons at that parent path. Evaluation comparisons
show task coverage and compare only shared tasks with compatible metrics;
different task sets never imply an overall rank. Training comparisons use
recorded training metrics. The public HTML report is `interactive`; there is
no new public `derived` directory.

Legacy flat runs can be migrated into experiment folders, one old run per
experiment. The migration copies canonical bytes first and preserves old
isolated exports under `isolated/<run-id>--legacy`; fresh isolated exports use
the ordinary `isolated/<run-id>` path. Historical cross-run grouped exports
are kept under `exports/legacy-comparisons/` without assigning them to a new
model or task group.

```bash
python -m ovlab_benchctl.experiment_migration preview --data-root /path/to/ovlab-data
python -m ovlab_benchctl.experiment_migration apply --data-root /path/to/ovlab-data
# Generate and verify the unified interactive reports listed in the manifest.
python -m ovlab_benchctl.experiment_migration finalize --data-root /path/to/ovlab-data
python -m ovlab_benchctl.experiment_migration cleanup --data-root /path/to/ovlab-data
```

`preview` is read-only. `apply` copies and verifies each run and export, writes
`experiment-migration-v1.json`, and leaves all original paths intact.
`finalize` verifies the new reports, then moves exact old paths to a reversible
quarantine. `cleanup` verifies again before removing the quarantine and empty
shared legacy folders. The manifest includes each absolute destination run
path, report requirement, source export paths and interactive output root.
