# OVLAB QuIC wrapper bones

Gate F defines two distinct, contract-complete integration identities:

- `quic-peft` is the published **Quantum-Inspired Compound Adapter** method. It
  applies a multiplicative adapter `W_adapt = ΔW_Q W*`, keeps the full frozen
  pretrained model, and reports base, adapter, trainable, and runtime resources
  separately. Adapter efficiency is not standalone deployed-model compression.
- `quic-wc` is a proposed dissertation extension for replacing selected dense
  weights with compact factors. It is not a result established by the original
  QuIC paper. Its factorization family remains deliberately unselected.

## Ownership boundary

Production dependency direction is strictly:

```text
OVLAB wrapper -> openvla_quic.ovlab_provider -> external QuIC implementation
```

`external/openvla-quic` must never import OVLAB internals. Its future public
provider returns JSON-compatible descriptions and NumPy action arrays through
these operations:

- `api_version()`
- `describe()`
- `capability_description()`
- `load(request)`
- `reset_episode(request)`
- `predict(request)`
- `load_counts()`
- `close()`

`describe()` must return the exact family, variant, scientific hash, base-model
identity, artifact identity, normalization identity, QP profile, and placement
manifest supplied by the OVLAB descriptor. `capability_description()` declares
concrete image/proprioception channels, the action specification, horizon, and
lifecycle capabilities. `load()` returns runtime/component identity and load
counts. OVLAB maps these neutral declarations into `PolicyCapabilities` and
passes only request/episode/step IDs, the authoritative instruction, and the
negotiated non-privileged observations to `predict()`.

Artifact and deployment evidence is variant-specific. PEFT uses a
`multiplicative_adapter` artifact and must state whether that adapter is active
and merged while retaining the base-model requirement. WC uses
`compact_weight_factors` and must prove replacement, absence of replaced dense
weights as a deployment requirement, and the prohibition on persistent dense
reconstruction. Provenance remains separate from artifact identity. All model
layers, compound matrices, factorization, injection, conversion, kernels,
training, and inference stay external.

## QP profiles

QP0 means no active QuIC transformation. QP1 through QP4 denote increasing
strength only within one mode; equal PEFT/WC labels are not equivalent. A future
runnable QP1–QP4 must carry a versioned definition and SHA-256, plus a versioned
placement manifest. Gate F assigns no numerical profile or placement values.

The QuIC paper's C1 relationship is to **Orthogonal Fine-Tuning** (Qiu et al.).
It is unrelated to OpenVLA-OFT, where OFT means **Optimized Fine-Tuning**.

## Gate boundary

Gate F proves descriptor, validation, hashing, capability, lifecycle, and failure
contracts only. Both variants report `implementation_status: skeleton`,
`runtime_validated: false`, and `compression_verified: false`. Loading fails with
`QuICImplementationUnavailableError` before provider discovery, CUDA, checkpoints,
socket readiness, or trace creation. It provides no evidence of correctness,
compression, latency, throughput, memory savings, or benchmark performance.
