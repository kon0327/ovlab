# Public verification examples

These examples exercise the supported configuration architecture. Personal
experiments and ablation studies belong in ignored `experiments/`, and personal
component overrides belong in ignored `configs/`. Plugin defaults and category
examples live in their individual projects under `plugins/`.

| Example | Coverage |
| --- | --- |
| [`mock-smoke.yaml`](experiments/mock-smoke.yaml) | Model-free descriptor planning and verification; mock execution uses the CPU test scaffold. |
| [`openvla-eval.yaml`](experiments/openvla-eval.yaml) | OpenVLA merged-LoRA checkpoint, one task, one two-step rollout, reporting. |
| [`openvla-oft-eval.yaml`](experiments/openvla-oft-eval.yaml) | OFT dual-camera/proprioception input, one task, one two-step rollout, reporting. |
| [`plugin-pipeline-smoke.yaml`](experiments/plugin-pipeline-smoke.yaml) | QuIC training, checkpoint finalization and handoff, evaluation, reporting. |

```bash
ovlab run examples/experiments/mock-smoke.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
```

The plugin pipeline example runs sequentially on one GPU with a 24 GiB resource
budget. It uses three optimizer steps and a small train/validation partition of
one LIBERO-10 task. Evaluation consumes the finalized adapter produced by that
same experiment; it does not fall back to the pretrained policy. Checkpoints and
datasets must be available through the normal OVLAB data preparation workflow,
and runtime images must be built from the registered plugins.

These limits test integration, not model quality. Reports record the training
and evaluation evidence even though a two-step rollout cannot establish useful
task performance. Copy an example into `experiments/` before creating a real
study, then select suitable data, episode limits and compute resources.

[`policy-plugin/`](policy-plugin/README.md) is the independent developer example
for implementing and registering another plugin. Model plugin category examples
describe local extensions; they are templates requiring appropriate checkpoints,
datasets and benchmark input contracts before execution.
