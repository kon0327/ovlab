# quic-petf-vla

Portable QuIC method extracted from `quic-petf-mpv/quic_openvla/adapter.py`.
The package name preserves the requested `petf` spelling. The Python API is
`quic_petf_vla`. It depends only on PyTorch and safetensors; OpenVLA, OVLAB and
benchmark dependencies belong to callers.

`inject(model, {"exact.module.name": AdapterConfig(...)})` freezes the host and
creates output-side determinant C1/C2/C3 adapters. `adapter_parameters(model)`
is the optimizer parameter list. Bias is frozen and unchanged. Only plain,
materialized, floating-point `torch.nn.Linear` layers are accepted.

`save_adapter(model, directory, host_identity, metadata=None)` writes
`adapter.json` and `adapter.safetensors`. Host identity must contain an immutable
SHA256 digest. Callers bind their preprocessing, normalization and data split in
host identity and metadata. `load_adapter(model, directory, host_identity)`
verifies the exact identity, target manifest, tensor digest, binding, shape and
dtype before changing the model. Existing artifacts cannot be overwritten.

`disabled_adapters(model)` selects the same frozen host for paired comparisons.
`merge_adapters(model)` is an in-memory dense layer merge, not a checkpoint
export or whole-model compression.
