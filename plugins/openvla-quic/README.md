# openvla-quic plugin

QuIC multiplicative-adapter inference and QuIC training. Dense inference and training only (`none`). Quantized hosts are unsupported.

## Registration and dependencies

```bash
ovlab plugin register /path/to/openvla-quic
ovlab plugin inspect quic-peft
```

`plugin.json` declares pinned upstream source commits, model capabilities,
build roles, checkpoint catalogs and optional training hooks. This project can
be registered independently; it does not require another model plugin.

Shared Python primitives live in the **ovlab-openvla-support** distribution.
The manifest explicitly declares it as the `openvla-support` source and each
build includes and installs it. The checked-in registration uses the local
`../_shared/openvla-support` development project. For an independently published
Git plugin, replace this source with a Git URL and full immutable commit; an
external local sibling is intentionally rejected for remote registrations.
Host configuration bundles freeze the declared control source for isolated use.

## Configuration categories

- `configs/policy/`: model runtime and interface defaults.
- `configs/deploy/`: persistent-service settings, separate from policy settings.
- `configs/registry/`: checkpoint links and identities; large weights stay outside Git/images.
- `configs/train/`: model training defaults, when supported; the experiment supplies its dataset.
- `configs/evaluate/`: model-side evaluation selection; the experiment supplies benchmark, metrics, protocol and artifact store.

Each supported category has one commented example under `examples/`.
Default references use `plugin://openvla-quic/CATEGORY/default.yaml`. Copy an example
to ignored `configs/`, extend the plugin reference, and explicitly select that
extension from your experiment. No local file is applied automatically.
Benchmark scenarios, smoke experiments and ablation studies belong to user
experiments or the small OVLAB verification examples, not this plugin.

The default policy is an adapter template. In a workflow, bind the finalized training checkpoint with `checkpoint_from`. Standalone evaluation/deployment requires an extension with the adapter path and both SHA256 identities; templates cannot execute directly.

Choose checkpoints explicitly through your policy/registry extension or bind a
training output from the experiment. `configs/registry/default.yaml` provides
available references; it contains no model weights. Changing quantization is a
`settings.runtime.quantization` override validated by the plugin, not a separate
copy of the policy config.

## Build and validation

```bash
ovlab plugin build quic-peft --dry-run
ovlab plugin build quic-peft
```

Builds acquire registered sources in isolation and need no mutable `external/`
checkout. Model-specific Docker recipes and dependency locks belong to this
project. Python packaging includes its manifest, defaults, examples, recipes,
locks and control sources; it does not include another model's integration.
