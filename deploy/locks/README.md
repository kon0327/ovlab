# Dependency locks

This directory owns infrastructure locks for the reporting image. Benchmark,
dataset and model runtime/training/finalizer locks live in their projects:

- [`plugins/libero/locks/`](../../plugins/libero/locks/) for benchmark and dataset.
- [`plugins/openvla/locks/`](../../plugins/openvla/locks/) for Vanilla and LoRA.
- [`plugins/openvla-oft/locks/`](../../plugins/openvla-oft/locks/) for native OFT.
- [`plugins/openvla-quic/locks/`](../../plugins/openvla-quic/locks/) for QuIC.

Each `*.in` describes a reviewed dependency set. Its PEP 751 `*.pylock.toml`
records immutable artifacts and hashes; the corresponding `*.requirements.txt`
is the hash-checked input consumed by Docker. Locks target the Python/platform
runtime named in their image recipe. Update a dependency deliberately, regenerate
the lock and rendered requirements, then validate the affected image's dependency
checks and integration smoke.

For example, using the target Python runtime:

```bash
python -m pip lock --output plugins/libero/locks/benchmark.pylock.toml -r plugins/libero/locks/benchmark.in
python deploy/scripts/render_requirements_lock.py plugins/libero/locks/benchmark.pylock.toml plugins/libero/locks/benchmark.requirements.txt
```

The benchmark closure explicitly selects headless runtime dependencies instead
of importing upstream training, notebook and device-tool extras. Its lock is
maintained with `--no-deps` when all transitive requirements are enumerated in the
input. Reporting has a separate offline renderer closure and contains no model or
simulator runtime.

Upstream source packages and local OVLAB/plugin distributions are installed with
dependency resolution disabled. Build recipes also disable unlocked PEP 517 build
isolation, preventing an upstream package from silently expanding the reviewed
closure. Source commits, lock digests and actual package inventories are recorded
in image provenance.
