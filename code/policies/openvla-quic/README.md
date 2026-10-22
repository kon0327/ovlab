# OVLAB QuIC

QuIC-PEFT is now an experimental adapter-only runtime for dense token OpenVLA.
Its method implementation lives in `external/quic-peft-vla`, independently
of OVLAB, OpenVLA and benchmarks. The OpenVLA fork imports that package; this
OVLAB package contains settings, provenance, lifecycle and policy contracts.

```text
OVLAB policy / training workflow
  -> external/openvla-quic (standalone dense model, loading, training/serving runtime)
    -> quic-petf-vla (compound math, injection, adapter artifacts)
```

The external fork also works independently: it exposes model-level QuIC APIs,
an explicit JSON training CLI, verified episode datasets, and HTTP serving.
Research-specific QP/A presets and placement selectors belong to
`ovlab_benchctl.quic_experiments`, not to the external model. The OVLAB protocol adapter lives in
`ovlab_openvla_quic.runtime`; it imports `quic_petf.host` only when loading a
policy. Descriptor validation, episode lifecycle and action conversions stay in
OVLAB. The external package contains no OVLAB provider or WC backend.

`external/compound-peft` remains an immutable historical reference. Its bundled
package named `peft` is never imported by the current runtime. The legacy
`quic-peft-bones.yaml` descriptor still represents its original unavailable
runtime. QuIC-WC remains a distinct, fail-closed proposed extension; it does not
reuse the dense adapter implementation and has no runnable policy.

## Initialize the external sources

```bash
git submodule update --init external/openvla-quic external/quic-peft-vla
```

Image builds use the checked-out source trees as they are. The submodule
revision is recorded by Git for provenance; build and deployment do not require
it to match a separate migration snapshot.

## Training and serving

Use the existing OVLAB training profile and adapter-only finalization workflow;
`configs/experiments/training/openvla-libero10-quic-smoke.yaml` is a short functional example.
It uses an explicitly identified dense host and episode-level train/validation
splits. `training.peft` selects placement, profile and explicit target selectors.
Only adapter parameters enter the optimizer; validation is teacher-forced
masked action-token CE and token accuracy, separate from rollout success.

`configs/policies/openvla-quic/adapter-libero10.yaml` and
`configs/experiments/evaluation/libero10-quic-adapter-rpc-smoke.yaml` demonstrate serving.
Set `settings.quic.adapter_path` to the finalized directory containing
`adapter.json` and `adapter.safetensors`; replace both zero SHA256 placeholders
with the hashes of those files. The zero values deliberately cannot load a real
artifact. `OVLAB_QUIC_ADAPTER_PATH` overrides only its execution location.
Docker additionally accepts `OVLAB_QUIC_ADAPTER_HOST_PATH` for the host mount.
The two content digests remain in scientific configuration identity when the
artifact is moved between machines or into the policy container.

Model and processor must use the same local checkpoint resource. The adapter
binds checkpoint bytes, immutable revision, model source, method source,
integration source, resolved targets, normalization and preprocessing. A changed
checkpoint, tokenizer, normalization, target manifest or source fails closed.
A previously saved standalone source artifact needs a matching host/source
identity and OVLAB metadata; renaming a directory does not migrate provenance.

The optional `--fit-action-normalization` training workflow fits q01/q99 from
training episodes only. Its artifact records `mode: train-q99`, the fitted
statistics and their SHA256 in `metadata.normalization_contract`; the same
digest is bound into host identity. Serving verifies that binding and installs
the statistics in memory before validating the normalization key, including
when that key does not exist in the base checkpoint. Host checkpoint files are
never modified. Without fitting, serving uses the original host statistics.

`OpenVlaQuICSettings` and `create_policy_adapter(settings)` use the same policy
lifecycle and RPC service as the existing OVLAB policies. The provider loads
once, accepts only the declared RGB channel and instruction, greedily predicts
one action, and releases model/processor references on close. No benchmark or
simulator is imported by the portable method or the OpenVLA provider.
The default decoded-action codec leaves source actions unchanged; the explicit
`openvla-to-libero-v1` OVLAB codec maps gripper convention exactly once through
a neutral threshold transform. Its action specification is declared through
the existing interface configuration, rather than hard-coded by the method.

## Evidence and supported scope

CPU checks cover C1/C2/C3 determinant values and gradients, Cayley orthogonality,
identity including BF16 bias rounding, frozen-host gradient flow, adapter
save/reload and tamper rejection, dense layer merge, plus a tiny explicit model
provider lifecycle. A bounded CUDA BF16 smoke verifies generic adapter forward,
backward, identity and exact save/reload on an actual GPU. These checks do not
validate 7B OpenVLA training, provider-to-standalone rollout parity or benchmark
quality; descriptor validation flags remain false until such evidence exists.

Supported: materialized plain `nn.Linear` and `nn.Conv2d`, output orientation,
adapter-only artifacts, unadapted or identified dense merged hosts. Conv2d
transforms output channels at each spatial position; grouped convolutions require
`block_count` divisible by `groups`. Mixed artifacts preserve convolution
geometry in format v2, while linear-only format v1 artifacts remain supported.
Unsupported: active LoRA,
quantized/specialized linears, OpenVLA-OFT, resumable training and full merged
checkpoint export. In-memory `merge_adapters` is not whole-model compression.
QP1–QP4 and `custom` carry resolved configuration hashes; QP0 remains solely the
legacy no-transformation descriptor. Shared labels do not make QuIC equivalent
to OpenVLA-OFT (Optimized Fine-Tuning). C1's orthogonal connection is to
Orthogonal Fine-Tuning.

## Optional action chunk variant

The implemented provider also loads continuous MLP chunk heads from complete
QuIC policy bundles. The artifact supplies the head configuration and weights;
capabilities advertise its actual horizon and predictions contain `[H,7]` actions.
The codec converts every gripper coordinate. Existing token artifacts retain
horizon 1. See [configuration, training and artifact contracts](../../../docs/QUIC_ACTION_CHUNKS.md).
