# openvla-quic

Dedicated Python environment for the standalone model package in
`external/openvla-quic`. It installs the fork's training, serving, and test
extras. Benchmark simulators such as LIBERO are intentionally separate optional
dependencies and are not part of this environment.

The supported baseline follows the fork's Docker image: Python 3.10, PyTorch
2.2.0, torchvision 0.17.0, and CUDA runtime 12.1. The Conda specification
installs Python through Conda and the CUDA-enabled PyTorch wheels through pip;
the Conda PyTorch 2.2.0 build can fail to import with newer MKL versions.
Remaining package requirements and optional extras are declared by the
OpenVLA-QuIC project's `pyproject.toml` and `requirements/` files.

Create the environment from the OVLAB repository root and install the current
working-tree checkout:

```bash
conda env create --file deploy/environments/openvla-quic/environment.yml
conda run -n openvla-quic python -m pip install \
  -e 'external/openvla-quic[train,serve,dev]'
```

For an existing environment, update the Conda base and reinstall the checkout:

```bash
conda env update --name openvla-quic \
  --file deploy/environments/openvla-quic/environment.yml --prune
conda run -n openvla-quic python -m pip install \
  -e 'external/openvla-quic[train,serve,dev]'
```

The `train` extra includes RLDS/OXE preparation and the preserved upstream
LoRA tuner. Prepared-NPZ QuIC training needs only the core package. `serve`
adds FastAPI and Uvicorn; `dev` adds the standalone tests and package checks.
FlashAttention is not installed because the standalone implementation uses
PyTorch SDPA by default.

Run the fork's focused tests and verify the dependency graph:

```bash
conda run -n openvla-quic python -m pytest external/openvla-quic/tests/quic
conda run -n openvla-quic python -m pip check
conda run -n openvla-quic openvla-quic --help
```

`snapshots/YYYY-MM-DD/` records the environment actually used for validation.
It is an informational environment snapshot; it does not lock or gate the
OpenVLA-QuIC source revision.
