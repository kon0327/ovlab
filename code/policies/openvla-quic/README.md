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

`external/openvla-quic` never imports OVLAB internals. Its public provider bones
return JSON-compatible descriptions and will eventually return NumPy actions through
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

The two external paths are deliberately separate:

```text
QuIC-PEFT provider -> compound_peft_bridge -> external/compound-peft
QuIC-WC provider   -> independent WC skeleton
```

The WC module does not import the compound-PEFT bridge and cannot inherit its
dense adapter forward path.

## User-supplied compound-PEFT source

`external/compound-peft` is an immutable user-supplied snapshot, not official
author-published code and not a scientific oracle. It has no Git revision. Its
identity is the extracted manifest SHA-256
`8084213849149a47f9bf84dd0c9220b319faf7df8dba39cdef3894e85e00f845`;
the user-declared archive SHA-256 is
`b024ba61b852d83beec631b724489b3bc3055c4a883f2df0c05b6c9857103e9a`.
The archive itself was not available locally, so its byte size and safety
inspection remain unavailable rather than guessed.

The 130-file Apache-2.0 snapshot contains PEFT `0.12.1.dev0`,
`PeftType.COMPOUND`, `CompoundConfig`, `CompoundModel`, and Linear/Conv2d
wrappers. It supplied no automated tests. Static in-memory compilation proves
syntax only. The dependency-light tester contains no Torch, Transformers, or
PEFT, so numerical CPU characterization is deferred to Gate I without changing
an accepted environment.

The source README citation differs from the paper identity used by OVLAB. OVLAB
records *QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient
Fine-Tuning*, Snehal Raj and Brian Coyle, arXiv:2502.06916. It does not copy the
README BibTeX as scientific provenance.

## Legacy configuration translation

Raw source fields are preserved beside canonical fields. In particular,
legacy `r` maps to `canonical.num_blocks`; it is not a LoRA rank. The block
dimension is derived from `target_output_dimension / num_blocks`, and no direct
equivalence between `r` and the paper's `b` is asserted. `comp_1`, `comp_2`, and
`comp_3` are the only supported orders.

The bridge also records mappings for compound operation, block sharing,
orthogonality enforcement, adapter-chain length/composition, learnable scaling,
and offset blocks. Permanent/max/average operations, multi-adapter chaining,
additive composition, learnable scaling, and offset blocks remain implementation
extensions unless separately validated against the paper.

Known limitations are explicit: the paper formulation assumes square matrices;
rectangular OpenVLA projections and target placement are unverified; the legacy
forward constructs a dense adapter; determinants are promoted to float32; and
merge/unmerge, checkpoint round-trip, and paper/forward/merged numerical
equivalence are implemented or claimed but unvalidated. Dense materialization is
permitted only for the PEFT reference and is forbidden for WC.

## QP profiles

QP0 means no active QuIC transformation. QP1 through QP4 denote increasing
strength only within one mode; equal PEFT/WC labels are not equivalent. A future
runnable QP1–QP4 must carry a versioned definition and SHA-256, plus a versioned
placement manifest. Gate F assigns no numerical profile or placement values.

The QuIC paper's C1 relationship is to **Orthogonal Fine-Tuning** (Qiu et al.).
It is unrelated to OpenVLA-OFT, where OFT means **Optimized Fine-Tuning**.

## Gate boundary

Gate F proves source intake, descriptor, validation, hashing, capability,
lifecycle, and failure contracts only. QuIC-PEFT reports source present and a
legacy reference backend available, but its OpenVLA integration remains a
skeleton. It raises `QuICPEFTIntegrationIncompleteError`. QuIC-WC has no source
and raises `QuICWCImplementationIncompleteError`. Both fail before provider
discovery, CUDA, checkpoints, socket readiness, or trace creation, and report
runtime/compression validation as false. Gate F provides no evidence of paper
equivalence, OpenVLA correctness, compression, latency, throughput, memory
savings, or benchmark performance.
