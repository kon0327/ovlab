# Experiments

Every supported experiment starts with the same command:

```bash
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
ovlab run examples/experiments/openvla-eval.yaml --verify
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
ovlab run EXPERIMENT_ID_OR_HASH --verify
ovlab report EXPERIMENT_ID_OR_HASH
```

The resolver accepts an explicit YAML/JSON path, a registered `plugin://` reference,
or a unique name below `experiments/` and `examples/experiments/`. Ambiguous names
require an explicit path. Tracked examples are verification scenarios; user
experiments live in the ignored `experiments/` directory. The document determines execution:

| Source | Interpretation |
| --- | --- |
| `kind: training`, `ovlab.training-profile/v1` | One root node executing training. |
| `kind: experiment`, schema `0.1.0` | One root node executing evaluation. |
| `kind: workflow`, `ovlab.experiment/v1` | A hierarchical tree of training and evaluation phases. |

All three forms share the experiment owner, metadata, phase storage and reporting.
The original source is preserved; standalone files are normalized to a one-node
tree for planning and execution. Phase-specific runtimes remain internal workers.

## Planning and verification

`--plan` resolves the complete configuration tree, validates profiles, component
references and dependencies, and displays the planned phases. It does not start
containers, initialize models or write run/export directories. Model/dataset
availability is checked by the phase runtime before execution.

`--verify` on a configuration performs the same descriptor validation. With an
existing instance ID, canonical path or suffix hash it verifies that experiment's
stored spec, plan, result and every recorded canonical training/evaluation run.
For example, `ovlab run 24265044c1 --verify` selects the experiment instance, while
a displayed eight-character run hash selects one run. Missing phases, changed
checksums, invalid metadata, symlinks and ambiguous references fail explicitly.
Verification is read-only. Reporting accepts the same experiment IDs and hashes.
`--plan` and `--verify` are mutually exclusive. `--detail` and `--json` control output.

## Hierarchy and execution settings

An `ovlab.experiment/v1` workflow defines nodes with stable IDs. A node can run
training, evaluation, both or neither. Grouping nodes contain children; phase
configuration inherits from ancestors. Mappings merge recursively. Lists and
scalar values replace their parents. `execute` explicitly schedules phases at
each node; it is not inherited. Trees are bounded to 12 levels and 512 nodes.

Scheduling belongs in the workflow configuration:

```yaml
execution:
  max_workers: 1
```

`1` executes sequentially; `2` through `32` permit that many concurrent phases.
The default is `2`. The scheduler does not allocate GPUs or reserve VRAM. Choose
a worker count that fits the configured device resources. Every configured phase
runs; use the tree's `execute` fields to define the desired scope.

A node's evaluation waits for its training. Explicit `depends_on` edges control
additional ordering. To evaluate a training output, declare its root-relative
node path in `config.evaluate.checkpoint_from`:

```yaml
config:
  evaluate:
    profile: plugin://openvla/evaluate/default.yaml
    checkpoint_from: lora
```

This selects the finalized checkpoint produced by the `lora` node's training in
the current experiment instance. The setting inherits into descendants, including
quantization variants, and automatically adds the training dependency. Use `""`
for training at the root. Missing or unselected sources, incompatible methods and
dependency cycles fail planning. An unsuccessful or unfinalized training phase
prevents evaluation; there is no fallback to the profile's reference weights.
Without `checkpoint_from`, an evaluation uses its explicitly configured model.

QuIC receives the newly finalized adapter path, hashes, base revision and
normalization contract. LoRA is merged with its verified base on CPU in the
offline `policy-openvla` image before inference quantization. Full, 8bit and 4bit
evaluations reuse one immutable dense snapshot, including the training
normalization and processor. Snapshots live under
`checkpoints/local/inference/<checkpoint-id>/<conversion-id>/`; the original
training bundle remains immutable. The merge image must expose the
`lora-merge-v1` checkpoint-preparation capability; build it with
`bash deploy/scripts/build-images.sh policy-openvla`.

Each evaluation result records `checkpoint_binding`: source node/run,
checkpoint ID, source manifest hash and inference identity. Its exact bound
configuration is saved in `.experiment/checkpoint-bindings/`. Experiment
verification checks this evidence against the canonical training result and
checkpoint. On failure, no new phases start; active subprocesses are interrupted
and completed evidence is retained.

The [pipeline example](../examples/experiments/plugin-pipeline-smoke.yaml) combines
QuIC training with checkpoint handoff, one two-step LIBERO rollout and reporting.
The [OpenVLA example](../examples/experiments/openvla-eval.yaml) and
[OFT example](../examples/experiments/openvla-oft-eval.yaml) verify pretrained
inference. The [mock configuration example](../examples/experiments/mock-smoke.yaml)
validates without model weights; it is a planning fixture, not a production runner. Copy an example into `experiments/` and explicitly change its
profiles and patches to define your experiment.

Running creates a new instance named with the configured ID, UTC timestamp and
ten-character suffix. `runs/<instance>/.experiment/` contains `spec.json`,
`plan.json`, `result.json` and the original `spec.yaml` when applicable. The plan
records resolved configurations and field provenance. The result records each
phase's run ID, exports and failures. Completed run evidence remains authoritative
if a later phase or reporting operation fails.

## Workflow fields

| Field | Purpose |
| --- | --- |
| `schema_version` | Required at the root; `ovlab.experiment/v1`. |
| `kind` | Required at the root; `workflow`. |
| `execution.max_workers` | Optional root scheduler limit, 1–32; default 2. |
| `id`, `name` | Portable ID and readable title. `id` is required at every node; `name` at the root. |
| `config.train.profile` | Training document path or `plugin://MODEL/train/default.yaml`. |
| `config.evaluate.profile` | Evaluation document path or `plugin://MODEL/evaluate/default.yaml`. |
| `config.evaluate.checkpoint_from` | Root-relative training node whose finalized output this evaluation uses; inherited by descendants. `""` selects the root. |
| `config.train.patch` | Changes fields in the resolved training profile, such as `dataset`, `model`, `training` or `resources`. |
| `config.evaluate.patch` | Changes fields in the resolved evaluation YAML, including `components` references and `deployment`. |
| `config.evaluate.component_patches` | Changes validated benchmark, policy, metrics, protocol or artifacts YAML; for example `benchmark.settings.task_indices`. |
| `execute` | Explicit array containing `train`, `evaluate` or both. This field is **not inherited**. |
| `depends_on` | Array of `{ "node": "root-relative/path", "phase": "train" }` scheduling dependencies. Use `""` for the root node. |
| `children` | Array of nested nodes; each child may override inherited configuration. |

Configuration objects inherit through the tree. Mappings merge recursively;
arrays and scalar values replace their parents. A child can select a new
`profile` and still receives inherited patches, so the resulting profile is
validated in full after composition. Plugin train/evaluate defaults can be partial:
the experiment must supply a dataset for training and benchmark, metric, protocol
and artifact choices for evaluation. A default is a model-owned building block,
not a complete experiment.

Configuration has three origins: `ovlab://` selects tracked `config/` defaults,
`plugin://` selects registered plugin defaults, and plain component references
select ignored user `configs/`. Relative `extends` stays inside its origin;
explicit URIs cross origins. No user override is discovered or applied implicitly.
Unknown fields, missing components, incompatible interfaces and dependency cycles
fail planning. See [configuration ownership](../config/README.md).

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
