# QuIC model plugin runtime

The independent [`openvla-quic` project](../../plugins/openvla-quic/README.md)
owns its policy, training and checkpoint-finalizer recipes. Its manifest declares
pinned upstream OpenVLA-QuIC and QuIC-PEFT sources plus the shared OpenVLA support
package. OVLAB acquires those dependencies through the same source interface as
any external plugin.

```bash
bash deploy/scripts/build-images.sh policy-openvla-quic training-openvla-quic finalizer-openvla-quic dataset benchmark reporting
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
```

Source synchronization into `external/` is optional for the default isolated
build. Existing local edits there are preserved. Plugin image settings can be
overridden through `OVLAB_QUIC_IMAGE`, `OVLAB_TRAINING_QUIC_IMAGE` and
`OVLAB_FINALIZER_QUIC_IMAGE`.

## Configuration and checkpoint handoff

Plugin defaults are modular:

```text
plugin://openvla-quic/policy/default.yaml
plugin://openvla-quic/deploy/default.yaml
plugin://openvla-quic/registry/default.yaml
plugin://openvla-quic/train/default.yaml
plugin://openvla-quic/evaluate/default.yaml
```

Train/evaluate defaults are fragments completed by an experiment. Task selection,
smoke limits and ablations belong to ignored user experiments, rather than the
plugin's shipped defaults. Each supported category has one commented example in
`plugins/openvla-quic/examples/`.

The policy default intentionally contains no fabricated adapter path or SHA.
`checkpoint_from` binds the finalized adapter from a selected training phase and
adds its execution dependency. A failed/unfinalized training output prevents
evaluation. An evaluation-only scenario must provide an existing adapter's real
path and verified manifest/tensor identities in its local policy extension.

Runtime validates and mounts the adapter directory read-only at
`/checkpoints/quic-adapter` and exposes that path through
`OVLAB_QUIC_ADAPTER_PATH`. The dense checkpoint uses the ordinary verified
checkpoint resolver. The benchmark receives neither model weights nor adapter
mounts. An explicit `OVLAB_QUIC_ADAPTER_HOST_PATH` override changes the
machine-local adapter location; it does not replace identity verification.

## Dependency and finalization boundaries

QuIC locks live in [`plugins/openvla-quic/locks/`](../../plugins/openvla-quic/locks/).
They select the standard Transformers runtime and exclude the OFT-specific
Transformers fork. Policy/training roles include the preprocessing and tensor
libraries they need; rollout environments remain in the benchmark image. Every
production build records inventories and runs dependency checks.

The trainer writes staged output. Its separate CPU finalizer verifies the
adapter inventory and scientific manifest before publishing the immutable
checkpoint. Rebuild the finalizer whenever checkpoint-format support changes;
it is not provided by the dataset image.

## Console progress

The shared console progress utility reports optimizer steps, validation batches
and evaluation episodes on stderr. Interactive terminals redraw a progress bar;
redirected runs emit periodic plain-text updates. JSON results remain on stdout.
`OVLAB_PROGRESS=on` records every update and `OVLAB_PROGRESS=off` silences progress
output. Persisted metrics and checkpoint evidence are independent of presentation.
