# OVLAB OpenVLA-OFT policy

This package is a thin OVLAB adapter around the pinned implementation in
`external/openvla-oft`. OFT means **Optimized Fine-Tuning**. The package does
not copy model layers or training code; it resolves and verifies the official
immutable checkpoint, translates negotiated OVLAB observations, invokes the
official inference function, and emits canonical LIBERO action chunks.

The Gate E reference uses the published merged runtime backbone. Its published
unmerged LoRA adapter is inventoried but is not active at runtime.

The backbone supports `quantization: none | 8bit | 4bit` for inference. The
8-bit mode uses BitsAndBytes LLM.int8 and the 4-bit mode uses NF4 with BF16
compute and double quantization. Auxiliary action-head and proprio-projector
components remain BF16. Runtime quantization is recorded separately from the
original OFT training and artifact identity.
