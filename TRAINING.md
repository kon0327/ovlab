# Training and checkpoint handoff

Model plugins own training implementations and defaults. OVLAB owns dataset and
checkpoint storage, scheduling, evidence and reporting. Training is selected
explicitly by an experiment; registering a model does not schedule training.

The maintained [pipeline smoke](examples/experiments/plugin-pipeline-smoke.yaml)
trains a QuIC adapter for three optimizer steps, finalizes its immutable
checkpoint, evaluates that checkpoint on one LIBERO task and publishes reports.
It is configured for a single 24 GiB GPU; actual resources and inputs are checked
by runtime preflight.

```bash
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
```

## Compose your own training

Copy the workflow into ignored `experiments/` and select a model's default:

```yaml
config:
  train:
    profile: plugin://openvla-quic/train/default.yaml
    patch:
      id: my-training
      dataset:
        ref: libero/libero_10
      training:
        normalization:
          unnorm_key: libero_10
```

This fragment illustrates composition, not a complete workflow. See each
plugin's `examples/train.yaml` and the pipeline example for checkpoint choice,
partitioning, resource limits and execution. Plugin defaults deliberately leave
`dataset.ref` to the experiment. Complete composed profiles use
`ovlab.training-profile/v1` and `kind: training`.

OpenVLA supports LoRA and full fine-tuning. OpenVLA-QuIC supports its own adapter
training. OFT currently supports inference only. Quantization and adaptation
support is declared in each plugin manifest and validated by that plugin;
unsupported combinations fail rather than silently selecting another method.
See [OpenVLA](plugins/openvla/README.md), [OFT](plugins/openvla-oft/README.md) and
[QuIC](plugins/openvla-quic/README.md).

Local extensions can `extends: plugin://openvla/train/default.yaml` and supply
only their changed fields. Select the extension explicitly in your workflow.
Ablations and dataset-specific recipes belong to your experiment/configuration
collection. OVLAB does not ship a matrix of historical research presets.
The generic `ovlab train ablate` commands still accept an explicitly selected
user suite; `ovlab train ablate --help` describes the commands.

## Data and execution

Use [dataset operations](DATASETS.md) to explicitly fetch/import and verify data.
The normal training pipeline is:

```text
verified dataset preparation -> complete profile validation -> offline trainer
-> checkpoint finalization -> explicit evaluation binding -> reports
```

The host model-data root defaults to `../ovlab-data` and can be changed with
`OVLAB_MODEL_DATA_ROOT`. Source code and checkpoints remain separate:

```text
ovlab-data/
├── datasets/              Imported and prepared dataset revisions
├── checkpoints/           Verified model and adapter artifacts
├── runs/<instance>/       Canonical train/evaluate evidence and experiment plan
└── exports/<instance>/    Derived reports and publication outputs
```

The trainer writes staged artifacts; the finalizer verifies them before
publishing an immutable checkpoint. Training output records the model revision,
processor, normalization, dataset split, trainable parameters and weight hashes.
A failed or incomplete checkpoint cannot satisfy an evaluation dependency.

Declare `config.evaluate.checkpoint_from` to consume a finalized output from the
same workflow. `""` selects root training; another node uses its root-relative
path. The dependency and binding are recorded in the experiment plan and evidence.
LoRA merging occurs before inference quantization; QuIC keeps its adapter and
verified dense base as separate artifacts.

```bash
ovlab checkpoint list
ovlab checkpoint list --source registry --offline
ovlab checkpoint verify CHECKPOINT_ID
ovlab run EXPERIMENT_ID_OR_HASH --verify
ovlab report EXPERIMENT_ID_OR_HASH
```

Plugin checkpoint catalogs contain pinned references and file identities, not
weight files. A user may select a plugin checkpoint, a local checkpoint override,
or a previously trained artifact. Large weights are neither committed to the
plugin project nor copied into every runtime image.
