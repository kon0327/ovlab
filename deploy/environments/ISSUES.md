# Environment Reproduction Issues

This document records the environment issues discovered while reproducing the
OpenVLA and OpenVLA-OFT smoke-test environments for OpenVLABenchmark (OVLAB).
It distinguishes immutable environment evidence from the operational steps
required to reconstruct a working environment.

## General rules

1. Treat `environment-full.yml` as an immutable audit snapshot produced by
   `conda env export`. Do not edit it to make environment creation succeed.
2. Use `environment-replay.yml` as the operational Conda recipe. Remove local
   editable packages and replace non-PyPI packages with reproducible source
   references where possible.
3. Install OVLAB source dependencies explicitly after creating the Conda
   environment. Their paths and installation modes belong in the replay or
   bootstrap script, not in the raw snapshot.
4. Keep policy inference environments separate from training/data-processing
   environments. The latter require TensorFlow, RLDS and `dlimp`; policy
   inference does not.
5. Validate the same capability as the original smoke test. An unrelated import
   failure must be recorded, but must not automatically invalidate a working
   inference environment.

## OpenVLA

### Issue 1: `dlimp==0.0.1` is not available on PyPI

#### Symptom

Environment creation fails during the pip phase:

```text
ERROR: Could not find a version that satisfies the requirement dlimp==0.0.1
ERROR: No matching distribution found for dlimp==0.0.1
```

#### Cause

The original environment contains `dlimp` installed directly from Git rather
than from PyPI:

```json
{
  "url": "https://github.com/moojink/dlimp_openvla",
  "vcs_info": {
    "commit_id": "040105d256bd28866cc6620621a3d5f7b6b91b46",
    "vcs": "git"
  }
}
```

A plain `pip freeze` or Conda export may reduce this source dependency to the
unresolvable distribution requirement `dlimp==0.0.1`.

#### Resolution

Replace the plain requirement in `environment-replay.yml` with the pinned VCS
reference:

```yaml
- dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46
```

Do not publish a private wheel merely to solve replay. A commit-pinned VCS
dependency preserves the actual origin sufficiently for the current snapshot.
A vendored clone or internally built wheel can be introduced later for fully
offline builds, but it must retain the upstream commit in its provenance.

### Issue 2: `openvla==0.0.3` is a local editable project

#### Symptom

Environment creation fails with:

```text
ERROR: Could not find a version that satisfies the requirement openvla==0.0.3
ERROR: No matching distribution found for openvla==0.0.3
```

#### Cause

The distribution was installed in editable mode from the local OVLAB clone:

```json
{
  "dir_info": {"editable": true},
  "url": "file:///home/kony/dissertation/ovlab/external/openvla"
}
```

`openvla==0.0.3` is therefore not a sufficient public package reference.

#### Resolution

Remove `openvla==0.0.3` from the pip subsection of
`environment-replay.yml`. After environment creation, install the checked-out
source explicitly:

```bash
conda run -n ovlab-verify-openvla \
  python -m pip install --no-deps -e "$PWD/external/openvla"
```

The external repository commit must be captured separately, for example in a
source manifest:

```bash
git -C external/openvla rev-parse HEAD
```

The future Docker build should copy or mount the pinned source tree and perform
the same explicit installation. It must not depend on the original absolute
host path.

### Issue 3: the distribution name is not the import package name

#### Symptom

`pip show openvla` succeeds, while this check fails:

```python
import openvla
```

#### Cause

The installed distribution is named `openvla`, but the relevant Python package
provided by this repository is `prismatic`. Python distribution names and
import package names are not required to match.

#### Resolution

Use the following import check:

```bash
conda run -n ovlab-verify-openvla python -c '
import torch
import prismatic

print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Prismatic/OpenVLA import: OK")
'
```

Do not add an `openvla` shim package solely to make `import openvla` succeed.

### Issue 4: `dlimp` import fails because of a protobuf API mismatch

#### Symptom

Importing `dlimp` reaches `tensorflow_metadata` and fails with:

```text
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

The failure is preceded by TensorFlow CUDA plugin warnings. Those warnings are
not the immediate cause of the exception.

#### Cause

Generated modules from the installed `tensorflow-metadata` expect
`google.protobuf.runtime_version`, but the environment contains a protobuf
release that does not provide that API. The same failure occurs in both the
original OpenVLA environment and the reconstructed environment.

This is therefore an existing limitation of the frozen environment, not a
replay regression.

#### Resolution and scope

Do not mutate the audit snapshot or silently upgrade protobuf in the inference
replay environment. Record the reconstructed OpenVLA environment as:

```yaml
scope: inference_only
validation:
  pip_check: pass
  prismatic_import: pass
  inference_smoke_test: pass
  dlimp_import: fail
  training_data_stack: unavailable
known_issue: tensorflow_metadata_protobuf_api_mismatch
```

The production OpenVLA policy image should omit `dlimp`, TensorFlow, RLDS and
other training/data dependencies unless inference demonstrably requires them.
Create and validate a separate training image before training or dataset
processing is brought into OVLAB scope.

### Issue 5: pip's local Conda build URL is not a source dependency

#### Symptom

Metadata for `pip==26.0.1` may contain a temporary local build URL such as:

```json
{"url": "file:///home/.../croot/pip_.../work"}
```

#### Cause

This is Conda package provenance from the machine where pip was built. It is
not a reusable project dependency.

#### Resolution

Keep pip as a normal Conda dependency when needed. Do not copy its `file://`
URL into the pip requirements or replay manifest. Avoid listing pip in both the
Conda dependency section and the nested pip subsection.

## OpenVLA-OFT

### Issue 1: LIBERO appears installed but cannot be imported

#### Symptom

The OFT smoke test fails while importing `GenerateConfig`:

```text
ModuleNotFoundError: No module named 'libero'
```

The failing import in OpenVLA-OFT is:

```python
from libero.libero import benchmark
```

At the same time, `pip list` reports an editable installation:

```text
libero  0.1.0  /home/kony/dissertation/ovlab/external/libero
```

#### Cause

`pip list` confirms installed distribution metadata, not importability. The
default modern PEP 660 editable installation does not expose LIBERO's nested
source-tree layout in the form expected by OpenVLA-OFT.

#### Confirmed resolution

Install LIBERO explicitly using Setuptools editable compatibility mode:

```bash
conda run -n ovlab-verify-openvla-oft \
  python -m pip uninstall -y libero

conda run -n ovlab-verify-openvla-oft \
  python -m pip install \
  --no-deps \
  --editable "$PWD/external/libero" \
  --config-settings editable_mode=compat
```

Then validate the actual import, not only package metadata:

```bash
conda run -n ovlab-verify-openvla-oft python -c '
from libero.libero import benchmark
print("LIBERO benchmark import: OK")
'
```

The compatibility installation was tested and resolves the import failure.
No `PYTHONPATH` override and no source modification are required.

#### Replay and Docker requirement

Remove the local editable `libero==0.1.0` entry from the pip subsection of
`environment-replay.yml`. Represent it as an explicit source dependency:

```yaml
source_dependencies:
  libero:
    path: external/libero
    editable: true
    editable_mode: compat
    install_dependencies: false
    reason: >-
      The default PEP 660 editable installation does not expose the nested
      package layout expected by OpenVLA-OFT.
```

The OFT replay/bootstrap script and future Dockerfile must use the same
`editable_mode=compat` installation. Capture the exact LIBERO commit in the
source manifest. Because compatibility mode is transitional Setuptools
behaviour, keep the import smoke test as a required regression check.

### Issue 2: OpenVLA-OFT is also a local editable project

The `openvla-oft` distribution is installed from:

```text
/home/kony/dissertation/ovlab/external/openvla-oft
```

Apply the same rule used for OpenVLA: do not expect the version string alone to
reconstruct the source. Remove the local editable entry from the replay pip
list, capture the repository commit, and install it explicitly after creating
the environment:

```bash
conda run -n ovlab-verify-openvla-oft \
  python -m pip install --no-deps -e "$PWD/external/openvla-oft"
```

Install source dependencies in deterministic order:

1. Create the Conda environment from `environment-replay.yml`.
2. Install LIBERO with `editable_mode=compat`.
3. Install OpenVLA-OFT as an editable local project.
4. Run import checks.
5. Run the frozen OFT action-chunk smoke test.

## Required verification matrix

Every replay and future image build should record these checks alongside the
environment snapshot:

| Environment | Check | Expected result |
| --- | --- | --- |
| OpenVLA | `python -m pip check` | Pass |
| OpenVLA | `import prismatic` | Pass |
| OpenVLA | CUDA availability | Pass on GPU runner |
| OpenVLA | Frozen inference smoke test | Pass |
| OpenVLA | `import dlimp` | Known failure in frozen full environment |
| OpenVLA-OFT | `python -m pip check` | Pass |
| OpenVLA-OFT | `from libero.libero import benchmark` | Pass |
| OpenVLA-OFT | OpenVLA-OFT project import | Pass |
| OpenVLA-OFT | Frozen action-chunk smoke test | Pass |

For each check, store the command, exit code, stdout/stderr, environment name,
Python version, GPU/runtime metadata, source commits and hashes of the smoke
test scripts and inputs.

## Instructions for future maintainers and Codex

- Never edit a raw `environment-full.yml` snapshot.
- Never treat a successful `pip list` entry as proof that a package imports.
- Never replace local editable source packages with unverified PyPI packages of
  the same name and version.
- Never copy machine-specific `file://` paths into replay or Docker dependency
  definitions.
- Keep source repository commits in a separate manifest and fail replay if a
  required checkout is missing or points to a different commit.
- Use `--no-deps` for explicit source installs after Conda/pip dependencies have
  been resolved from the replay file; this prevents local project metadata from
  unexpectedly changing the frozen dependency graph.
- Preserve the distinction between an inference-only success and a fully
  functional training/data environment.
- Re-run the complete verification matrix whenever Python, pip, Setuptools,
  protobuf, TensorFlow, LIBERO, OpenVLA or OpenVLA-OFT changes.
- In Docker, use container paths rather than WSL host paths. Do not bake
  `/home/kony/...` into images or configs.
- Keep Vanilla/LoRA, OFT and QuIC policy environments isolated. Do not merge
  their dependency graphs merely because they share the OpenVLA lineage.

