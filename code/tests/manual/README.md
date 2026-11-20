# Manual model validation

These tests load real model weights or run LIBERO/GPU inference. They require an
explicit prepared model environment, datasets, checkpoints and opt-in flags from
the selected test module. They are separate from the automated CPU/RPC suite.

Historical research presets are no longer shipped. Supply a complete standalone
evaluation document through `OVLAB_MANUAL_EVALUATION_CONFIG` and machine paths
through `OVLAB_LOCAL_PROFILE`. The evaluation must select the method and input
contract expected by the selected test: merged LoRA, native OFT, or the LIBERO
mock policy as documented by that module. Local files belong under ignored
`configs/` or `experiments/`; extend plugin defaults and select compatible OVLAB
components explicitly. The maintained workflow examples illustrate composition,
but a low-level resolver test needs a standalone `kind: experiment` document.

Use `ovlab run TARGET --plan` first. Run these files individually with their
required environment and flags; they deliberately reuse descriptive test module
names. See [plugin configuration](../../../docs/POLICY_PLUGINS.md) and
[training](../../../TRAINING.md) for the current architecture.
