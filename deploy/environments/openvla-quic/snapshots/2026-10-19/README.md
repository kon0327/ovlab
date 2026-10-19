# openvla-quic environment snapshot

Captured 2026-10-19 from the dedicated Conda environment at
`/home/kony/miniconda3/envs/openvla-quic`, after installing the current local
`external/openvla-quic` checkout with its `train`, `serve`, and `dev` extras.
PyTorch, torchvision, CUDA, and the method package are recorded by
`runtime.txt`, `pip-freeze-all.txt`, and `pip-inspect.json`.

This snapshot records the packages used for the tests. It is informational: it
does not enforce a model or external-source revision. The portable base recipe
is `../../environment.yml`; after creating it, install the current checkout as
described in `../../README.md`.

See `test-results.txt` for the test commands and outcomes. CUDA was not
available inside the test process, so CUDA-only coverage was skipped.
