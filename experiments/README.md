# Experiment directories

Experiment scenarios live separately from the reusable components in
[`configs/`](../configs/DIRECTORIES.md). Paths below are relative to this directory.

```text
experiments/
├── evaluation/            Evaluation scenarios
├── training/              Training profiles
│   └── quic/              QuIC training variants
│       ├── action-chunks/ Action-chunk experiments
│       └── lora-coverage/ LoRA module-coverage recipes
└── workflows/             Hierarchical experiment specifications
```

- **`evaluation/`** defines scenarios selecting a benchmark, policy, metric set,
  protocol, artifact store and resource registry, with optional deployment and
  reporting. It includes smoke, qualification and LIBERO comparisons. Run with
  `ovlab run TARGET`.
- **`training/`** contains model, dataset, optimization and resource profiles,
  including reference and smoke recipes. Use `ovlab run TARGET` or
  `ovlab run TARGET --verify`.
  - **`quic/`** holds QuIC variants, including placement variants used by the
    [ablation suite](../docs/ABLATIONS.md).
  - **`quic/action-chunks/`** compares token and action-chunk heads; see
    [action chunk contracts](../docs/QUIC_ACTION_CHUNKS.md).
  - **`quic/lora-coverage/`** contains smoke and longer LoRA coverage recipes
    for LIBERO-10, Goal, Object and Spatial; see the
    [coverage guide](../docs/QUIC_LORA_COVERAGE.md).
- **`workflows/`** contains YAML or JSON hierarchical specifications combining
  training and evaluation profiles, patches and dependencies. Use
  `ovlab run --plan FILE` or `ovlab run FILE`; see the
  [workflow guide](../docs/EXPERIMENT_WORKFLOW.md).

  - [`libero10-quic-training-smoke.yaml`](workflows/libero10-quic-training-smoke.yaml)
    trains QuIC for three steps on one task and evaluates its own checkpoint for
    one task with a two-step episode limit.
  - [`libero10-two-task-evaluation-only.yaml`](workflows/libero10-two-task-evaluation-only.yaml)
    evaluates LoRA, QuIC and OFT on two tasks without training. It selects the
    verified local QuIC adapter by path and hashes; update those values to use a
    different saved adapter.
  - [`libero10-reporting-eval-smoke.yaml`](workflows/libero10-reporting-eval-smoke.yaml)
    compares pretrained LoRA, OFT and Vanilla, each unquantized and 4-bit, on
    two tasks. Its hierarchy produces model and quantization comparison reports.
  - [`libero-object-quic-reporting-smoke.yaml`](workflows/libero-object-quic-reporting-smoke.yaml)
    trains QuIC on two LIBERO-Object tasks and evaluates its own checkpoint on
    those tasks. Both reporting smokes use one rollout per task capped at 30
    steps and run sequentially on one GPU; they do not measure model quality.

All three directories use `ovlab run EXPERIMENT`; their names organize reusable
source files and do not select different CLI commands. A unique YAML name can be
used directly; ambiguous names require a path. Add `--plan` to resolve the full
tree, or `--verify` to validate a source or check a stored experiment ID/hash.

Use the local [Workflow Studio](../code/apps/workflow-builder/index.html) to
assemble nodes visually from these profiles and export a workflow YAML. Run
`ovlab editor` to open it; see the [builder guide](../code/apps/workflow-builder/README.md)
for catalog refresh, inheritance and validation details.

Standalone training and evaluation sources become one-node experiments. Workflows
define their hierarchy explicitly with `children`; their configuration inherits
downward. Configure scheduling with root `execution.max_workers`. Every execution
uses the same `.experiment` metadata and canonical phase hierarchy.

The `example.yaml` files in each directory are commented reference documents.
Scenario inheritance uses explicit relative `extends` paths: mappings merge
deeply, while scalars and sequences replace their parent values.

Workflow execution may create temporary `.generated/` directories under
`training/`, `evaluation/` and component directories in `configs/`. These are
normally removed after the run; the resolved documents are recorded in the
immutable plan and run evidence.
