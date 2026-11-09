# Experiment workflow

An `ovlab.experiment/v1` JSON document describes one experiment tree. Any
node can run training, evaluation, both, or neither. Nodes without `execute`
group their children. The tree can be as deep as needed within the validated
limit of 12 levels and 512 nodes.

The [model comparison example](../configs/experiments/workflows/libero-model-tree.json)
has model branches, a task branch and quantization ablations. It also runs
training at the LoRA model node. Its evaluation profile points to a configured
checkpoint independently of that training run. `depends_on` controls execution
order only; it does not select a newly trained checkpoint. To evaluate the
checkpoint from a training run, configure a matching policy/checkpoint artifact
explicitly before launching evaluation.

## Plan and run

```bash
./ovlab experiment plan --spec configs/experiments/workflows/libero-model-tree.json --json
./ovlab experiment run --spec configs/experiments/workflows/libero-model-tree.json --dry-run
./ovlab experiment run --spec configs/experiments/workflows/libero-model-tree.json
./ovlab experiment run --spec configs/experiments/workflows/libero-model-tree.json --node lora/task-0
./ovlab experiment run --spec configs/experiments/workflows/libero-model-tree.json --node lora --node-only
```

`--node` selects that node and its descendants. `--node-only` selects just
that node. A selected subtree must contain every phase it names in
`depends_on`; otherwise planning fails before any workload starts. `plan` and
`run --dry-run` validate the selected subtree and write nothing. The JSON tree
shape is always checked, while unrelated siblings need not have runnable
profiles when a node is selected. Running creates
a new instance whose directory name contains the spec ID, UTC timestamp and
short content identity.

The plan contains each node's complete resolved train/evaluate configuration
and a source for every field. It is saved with the original JSON under
`runs/<instance>/.experiment/`. `result.json` records each phase's run ID,
canonical path, CLI result, comparison report and any failure. The export root
has an `experiment.json` index. Completed runs remain authoritative if a later
node or report fails.

## JSON fields

| Field | Purpose |
| --- | --- |
| `schema_version` | Required at the root; `ovlab.experiment/v1`. |
| `id`, `name` | Portable ID and readable title. `id` is required at every node; `name` at the root. |
| `config.train.profile` | Existing YAML under `configs/experiments/training/`. |
| `config.evaluate.profile` | Existing YAML under `configs/experiments/evaluation/`. |
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

Each phase uses the existing training or evaluation owner. For execution, the
workflow temporarily materializes the resolved YAML under the appropriate
`configs/**/.generated/` directory, then removes it. The immutable plan and
canonical run evidence retain the exact resolved documents.

## Artifacts and comparisons

```text
runs/<instance>/
  .experiment/{spec,plan,result}.json
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
