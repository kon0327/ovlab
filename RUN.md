# Running experiments

OVLAB composes model plugins with explicit benchmark, protocol, metric and report
choices. Start with a tracked verification example:

```bash
ovlab run examples/experiments/mock-smoke.yaml --plan
ovlab run examples/experiments/openvla-eval.yaml --plan
ovlab run examples/experiments/openvla-oft-eval.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
```

Remove `--plan` to execute a model example. The mock document is a configuration
verification fixture; production execution currently requires the LIBERO runtime.
The QuIC pipeline trains an adapter, finalizes its
checkpoint, evaluates that exact checkpoint and publishes reports. Its short
limits verify the integration rather than model quality. Runtime preflight still
checks images, GPU capacity, checkpoint and dataset availability.

Copy an example into `experiments/` to create your own scenario. Contents of
`experiments/` and `configs/` are ignored by Git; OVLAB ships only their READMEs.
Local edits therefore do not become upstream presets. Save explicit references
in your scenario; OVLAB does not automatically apply files from either directory.

## Configuration ownership

| Reference | Owner |
| --- | --- |
| `ovlab://metrics/default.yaml` | Builtin OVLAB configuration under `config/`. |
| `plugin://openvla/policy/default.yaml` | Configuration of a registered model plugin. |
| `policies/my-model.yaml` | User configuration under `configs/`. |
| `experiments/my-study.yaml` | User experiment, selected by its explicit path. |

A local policy can extend a plugin default and override only its runtime:

```yaml
schema_version: "0.1.0"
kind: policy
type: openvla_lora_merged
extends: plugin://openvla/policy/default.yaml
settings:
  runtime:
    quantization: 4bit
```

Save it as `configs/policies/my-model.yaml`, then explicitly select
`policies/my-model.yaml` in the experiment's `components.policy` or workflow
`config.evaluate.patch.components.policy`. Relative `extends` stays inside its
own configuration origin; explicit URIs cross origins. Mappings merge, lists and
scalars replace. The final composed documents are validated before execution.

Plugin evaluation defaults contain the model portion of an experiment. Your
scenario supplies benchmark, metrics, protocol and artifact configuration.
Training defaults likewise require an explicit dataset. Plugin defaults do not
silently select a scientific experiment.

## Execution and evidence

`ovlab run TARGET` accepts a training document, an evaluation document, or a
hierarchical workflow. Workflows can reference plugin training/evaluation defaults
directly. `execution.max_workers: 1` serializes phases on a single GPU.
`config.evaluate.checkpoint_from` binds a finalized training output and adds the
training dependency; `""` refers to root training.

Evaluation declares `deployment.profile` and `deployment.renderer`, starts policy
and benchmark services in separate containers, and cleans up its private Compose
project. Persistent model services use `ovlab deploy`; see
[DEPLOYMENT.md](DEPLOYMENT.md). Machine paths belong in `deploy/compose/.env` or an
ignored local profile selected by `OVLAB_LOCAL_PROFILE`.

```bash
ovlab data list runs --detail
ovlab run EXPERIMENT_ID_OR_HASH --verify
ovlab report EXPERIMENT_ID_OR_HASH
```

Canonical evidence lives under `ovlab-data/runs/`; reports live under
`ovlab-data/exports/`. Resolved configuration bundles retain their source identities
and are mounted read-only. Configuration changes do not modify earlier evidence.
See [the workflow contract](docs/EXPERIMENT_WORKFLOW.md),
[plugin development](docs/POLICY_PLUGINS.md), and [dataset operations](DATASETS.md).

## Optional Weights & Biases mirror

Set `OVLAB_WANDB_PROJECT` (or `WANDB_PROJECT`) and optionally `OVLAB_WANDB_ENTITY`.
Authenticate with `wandb login` or `WANDB_API_KEY`. The host mirrors completed
phases after isolated execution; W&B failures do not change canonical evidence.
