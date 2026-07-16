# ovlab-tester

`ovlab-tester` is the lightweight default environment for OVLAB's CPU-only unit, contract, and integration tests. It is intentionally independent of the large and mutually incompatible policy environments.

The environment contains only the common test runtime:

- Python 3.10,
- NumPy,
- pytest,
- pip and setuptools for package/build checks.

Policy-specific tests that require CUDA, model checkpoints, LIBERO, OpenVLA, OFT, QuIC, or their native dependencies must remain explicit manual or service-level tests and must use their corresponding environment.

## Create or update

From the OVLAB repository root:

```bash
conda env create --file deploy/environments/ovlab-tester/environment.yml
```

For an existing environment:

```bash
conda env update --name ovlab-tester \
  --file deploy/environments/ovlab-tester/environment.yml \
  --prune
```

## Run tests

The shared runner adds OVLAB's local `src/` package directories to `PYTHONPATH`; it never adds `external/`:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh
```

Pass normal pytest arguments to narrow the run or change verbosity:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/unit/core/contracts -q
```

Local packages are tested from the working tree rather than installed into the environment. This keeps the test result aligned with the currently checked-out OVLAB source and avoids stale editable-install paths.
