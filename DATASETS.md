# Dataset operations

OVLAB treats training datasets as immutable, verified inputs stored outside the
source repository. Dataset selection is expressed by a logical provider/name
pair; portable experiment and training configuration must not contain absolute
host paths.

The normal lifecycle is:

```text
resolve identity -> explicitly acquire or import -> verify -> use read-only
```

Dataset commands run in the dedicated `ovlab-dataset` image. They do not
require an activated Conda environment. Read-only commands run with
`--network none`; only the explicit `dataset fetch` operation receives network
access.

## Storage location

The default model-data root is `../ovlab-data` relative to the repository:

```text
ovlab-data/
└── datasets/
    └── <provider>/<name>/<version>/
        ├── raw/
        ├── prepared/
        ├── dataset.json
        └── manifest.json
```

Override the complete model-data root when necessary:

```bash
export OVLAB_MODEL_DATA_ROOT=/mnt/ovlab-data
```

For example, the pinned LIBERO-10 snapshot is published at
`datasets/libero/libero_10/1.0.0`. The readable version directory is not the
complete identity: `manifest.json` retains the exact provider revision,
content-derived dataset ID, preparation recipe and per-file SHA-256 values.
Publishing different bytes or a different source/preparation identity under an
existing version is rejected; select a new version instead.

Stores created with the former `<source-revision>/<build-id>` layout remain
readable during migration, but all new publications use the version directory.

The host path is execution provenance only. Moving an intact dataset store does
not change dataset scientific identity. Ready datasets are content-verified and
read-only; do not edit files below a published version.

Build the dataset image once after changing OVLAB source or dependency locks:

```bash
bash deploy/scripts/build-images.sh dataset
```

The image can be overridden explicitly with `OVLAB_DATASET_IMAGE`. Direct host
execution through `OVLAB_DATASET_RUNTIME=host` is a diagnostic escape hatch,
not the normal workflow.

## Discover and resolve datasets

List the installed provider bridges:

```bash
./ovlab dataset providers
```

Resolve a known LIBERO suite without downloading anything:

```bash
./ovlab dataset resolve --benchmark libero --suite libero_10
```

Supported built-in suite selectors are:

| Selector | OpenVLA RLDS dataset |
|---|---|
| `libero_spatial` | `libero_spatial_no_noops/1.0.0` |
| `libero_object` | `libero_object_no_noops/1.0.0` |
| `libero_goal` | `libero_goal_no_noops/1.0.0` |
| `libero_10` | `libero_10_no_noops/1.0.0` |

Resolution reports the pinned repository revision, bridge version, canonical
source and a `dsr-...` resolution ID. A resolution is not a ready dataset: it
does not allocate storage, contact the provider, or produce a `dataset-...` ID.

Add `--json` to any command for the stable `ovlab-cli-output/1.0.0` envelope:

```bash
./ovlab dataset resolve --benchmark libero --suite libero_10 --json
```

## Explicit LIBERO acquisition

Acquire and prepare a known suite explicitly:

```bash
./ovlab dataset fetch --source libero --name libero_10
```

This is the operation that may contact the pinned provider. OVLAB prints
per-file progress to stderr, downloads into a staging directory, validates
provider file digests where available, inventories every file, prepares the
OpenVLA RLDS layout, and publishes the manifest last. A completed result
contains an immutable `dataset-...` ID.

Repeating the same command reuses and re-verifies the existing content instead
of publishing a mutable replacement.

Dataset acquisition is not performed by `train validate` or `train plan`.
Training can opt into the same explicit built-in acquisition step with:

```bash
./ovlab train run \
  --profile configs/training/openvla-libero10-lora-smoke.yaml \
  --allow-dataset-download
```

Without that flag, a missing dataset stops before model initialization or
training-container startup.

## Acquire a verified URL dataset

URL acquisition requires an HTTPS URL and the expected SHA-256 of the
downloaded payload:

```bash
./ovlab dataset fetch \
  --source url \
  --name custom-robot-data \
  --version 1 \
  --url https://data.example.org/custom-robot-data.tar.zst \
  --sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --archive tar.zst \
  --format directory-v1
```

Supported archive modes are `auto`, `none`, `zip`, `tar`, `tar.gz`, `tgz` and
`tar.zst`. Use `none` for a single uncompressed payload. The SHA-256 is required
even when transport uses HTTPS.

OVLAB rejects:

- credentials embedded in URLs;
- non-HTTPS remote origins and unsafe redirects;
- checksum mismatches;
- absolute or parent-traversal archive paths;
- archive symlinks, hard links, devices and FIFOs;
- excessive file counts or extracted/downloaded size;
- an empty prepared dataset.

Failed or interrupted acquisition never enters the ready registry. Diagnostic
state remains below `datasets/.staging/`; retry the same explicit command after
inspecting the failure. For a pinned LIBERO snapshot, OVLAB hashes and reuses
complete staged files and downloads only missing or invalid files. A retry never
treats partial bytes as a finalized dataset.

## Import an existing local dataset

Import local data by copying it into controlled OVLAB storage:

```bash
./ovlab dataset import \
  --name quic-libero10-v1 \
  --version 1 \
  --path /absolute/path/to/source-dataset \
  --format directory-v1
```

The source directory is mounted read-only into the dataset container. OVLAB
rejects symlinks and copies regular files before computing the immutable
identity, so later changes at the original path cannot silently change a
training input. The original absolute path is not part of scientific identity
or recorded as the dataset locator.

Import does not use network access. A changed source produces a different
content-derived dataset identity rather than modifying the previous build.

## List, inspect, prepare and verify

List all locally ready datasets:

```bash
./ovlab dataset list
```

The default output is one readable row per dataset:

```text
libero_10 1.0.0 /home/user/ovlab-data/datasets/libero/libero_10/1.0.0
```

Print the complete dataset-list document for interactive inspection with
`./ovlab dataset list --detail`. For scripts, `--json` retains the stable
`ovlab-cli-output/1.0.0` envelope.

Inspect one manifest:

```bash
./ovlab dataset inspect --dataset dataset-0123456789abcdef0123456789abcdef
```

Important manifest fields include:

- provider, logical name and source revision;
- resolution and dataset IDs;
- raw and prepared aggregate digests;
- per-file paths, sizes and SHA-256 values;
- preparation recipe and bridge versions;
- schema, sample count, license and citation when known;
- acquisition state and sanitized source provenance.

Verify all recorded files without network access:

```bash
./ovlab dataset verify --dataset dataset-0123456789abcdef0123456789abcdef
```

Verification fails if a file is missing, mutable through a symlink, has a
different size, or does not match its recorded SHA-256. It does not repair or
replace corrupted content.

`prepare` confirms that an already-published immutable preparation has the
requested format and re-verifies it:

```bash
./ovlab dataset prepare \
  --dataset dataset-0123456789abcdef0123456789abcdef \
  --format openvla-rlds
```

It does not transform a published dataset in place. A different preparation
recipe must create a new immutable build.

## Training profile references

Portable training profiles use logical dataset references:

```yaml
dataset:
  ref: libero/libero_10
  preparation: openvla-rlds
  split: train
```

`train plan` resolves this selector to one verified local `dataset-...` ID and
includes the source revision, raw digest, preparation recipe/version and
prepared digest in scientific identity. Host paths, Docker tags and timestamps
remain execution provenance.

The training container receives only the resolved `prepared/` directory,
mounted read-only. It cannot access the acquisition staging area or modify the
canonical dataset.

## Operational guidance

- Keep `ovlab-data/datasets` outside the Git checkout and back it up together
  with its manifests. When copying, preserve permissions and hard links where
  possible (for example, `rsync -aH`).
- Use `dataset verify` after restoring or moving the store and before a long
  training run.
- Do not use `chmod`, editors or data-cleaning scripts inside a published
  dataset directory. Import corrected bytes as a new version instead.
- There is intentionally no CLI command that mutates or deletes a finalized
  dataset. Administrative removal must first establish that no canonical
  training run references the dataset ID.
- Never place credentials in YAML, URLs, command history or dataset metadata.

## Troubleshooting

`dataset image is unavailable`
: Build it with `bash deploy/scripts/build-images.sh dataset`, or select a
  verified image using `OVLAB_DATASET_IMAGE`.

`dataset_unavailable`
: Run `dataset list`, resolve the intended selector, and explicitly fetch or
  import it. Check that `OVLAB_MODEL_DATA_ROOT` points to the expected store.

`artifact_integrity_error`
: The acquired bytes or a published file do not match their manifest. Do not
  bypass verification. Preserve the failing evidence and reacquire/import into
  a new build.

`Permission denied`
: Check ownership and group access on `OVLAB_MODEL_DATA_ROOT/datasets`. The
  launcher runs the container as a non-root user and adds the caller's primary
  group; it deliberately does not use a privileged container.

For the subsequent profile, training, checkpoint and deployment workflow, see
[`TRAINING.md`](TRAINING.md).
