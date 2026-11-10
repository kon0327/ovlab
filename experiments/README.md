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
  `ovlab eval run TARGET`.
- **`training/`** contains model, dataset, optimization and resource profiles,
  including reference and smoke recipes. Use `ovlab train run TARGET` or
  `ovlab train validate TARGET`.
  - **`quic/`** holds QuIC variants, including placement variants used by the
    [ablation suite](../docs/ABLATIONS.md).
  - **`quic/action-chunks/`** compares token and action-chunk heads; see
    [action chunk contracts](../docs/QUIC_ACTION_CHUNKS.md).
  - **`quic/lora-coverage/`** contains smoke and longer LoRA coverage recipes
    for LIBERO-10, Goal, Object and Spatial; see the
    [coverage guide](../docs/QUIC_LORA_COVERAGE.md).
- **`workflows/`** contains YAML or JSON hierarchical specifications combining
  training and evaluation profiles, patches and dependencies. Use
  `ovlab experiment plan FILE` or `ovlab experiment run FILE`; see the
  [workflow guide](../docs/EXPERIMENT_WORKFLOW.md).

The `example.yaml` files in each directory are commented reference documents.
Scenario inheritance uses explicit relative `extends` paths: mappings merge
deeply, while scalars and sequences replace their parent values.

Workflow execution may create temporary `.generated/` directories under
`training/`, `evaluation/` and component directories in `configs/`. These are
normally removed after the run; the resolved documents are recorded in the
immutable plan and run evidence.
