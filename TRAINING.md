# Training operations

Training scenarios live in `configs/experiments/training/`. Use
`ovlab train run TARGET`, `ovlab train inspect RUN_REFERENCE`, and
`ovlab train verify RUN_REFERENCE`. `train run TARGET --dry-run` resolves the
training plan without starting a trainer; resource planning may require prepared
local datasets/checkpoints. The legacy `--run` form for inspect/verify remains
accepted. Evaluation uses `ovlab eval`; persistent model services use `ovlab deploy`.


Detailed dataset-provider, acquisition, local-import and integrity procedures
are documented separately in [`DATASETS.md`](DATASETS.md). This guide focuses
on the end-to-end training and checkpoint lifecycle.

OVLAB manages training as a separate, reproducible pipeline:

```text
explicit dataset acquisition -> verified preparation -> validated profile
-> offline trainer -> finalized immutable checkpoint -> explicit deployment handoff
```

Training never runs as part of `eval run`. Benchmark `runs/`, reporting
`exports/derived/` and other publication outputs remain separate from model data. The host
model-data root defaults to `../ovlab-data` relative to the repository and may be
set with `OVLAB_MODEL_DATA_ROOT`.

## Storage

```text
ovlab-data/
├── datasets/<provider>/<name>/<revision>/<build-id>/
│   ├── raw/
│   ├── prepared/
│   ├── dataset.json
│   └── manifest.json
├── runs/train/<training-run-id>/
│   ├── original-profile.yaml
│   ├── resolved-profile.yaml
│   ├── training-plan.json
│   ├── provenance.json
│   ├── events.jsonl
│   ├── metrics.jsonl
│   ├── data-flow.json              # bounded first-batch input/output evidence
│   ├── logs/
│   ├── staging-checkpoints/
│   ├── result.json
│   └── manifest.json
└── checkpoints/<immutable-checkpoint-id>/
    ├── weights-or-adapter/
    ├── processor/
    ├── auxiliary/
    ├── checkpoint.json
    └── manifest.json
```

Ready datasets and finalized checkpoints are read-only and content-addressed.
Evaluation evidence is stored alongside it under `runs/evaluate/<run-id>/`.
Existing training artifacts under `training-runs/` remain readable.
Host paths and timestamps are provenance only; they do not affect scientific
identities. Failed or interrupted work remains evidence under staging or the
training run and is never advertised as ready.

New OpenVLA runs capture the first actual microbatch and its model output in
`data-flow.json`. It records full tensor shapes/dtypes and bounded previews of
the first sample, including prompt/action tokens, transformed pixels, predicted
supervised tokens, loss and optimizer defaults. `metrics.jsonl` records actual
examples, microbatches, non-padding tokens and action-token accuracy per
optimizer step. These artifacts enter the finalized training inventory.
`train report` publishes the existing HTML and its interactive companion under
`exports/interactive/training/`, alongside `exports/derived/`; see [`REPORTING.md`](REPORTING.md).

The training CLI accepts custom images through `OVLAB_TRAINING_IMAGE` or
`OVLAB_TRAINING_QUIC_IMAGE`. Image preflight checks the declared training
deployment contract, not a fixed image role, Git revision or source hash.
Dataset and reporting images are checked against their respective contracts.
Rebuild an image when you want current source changes inside its container;
build provenance remains recorded in the resulting image.

## Dataset commands

Dataset commands use the dedicated non-root `ovlab-dataset` image. Read-only
operations have no network. Only `dataset fetch` receives network access and a
writable dataset mount.

```bash
./ovlab dataset providers
./ovlab dataset resolve --benchmark libero --suite libero_10
./ovlab dataset fetch --source libero --name libero_10
./ovlab dataset fetch --source url --name custom --version 1 \
  --url https://example.org/dataset.tar.zst --sha256 SHA256 --archive tar.zst
./ovlab dataset import --name custom --version 1 --path /host/source
./ovlab dataset list
./ovlab dataset inspect --dataset DATASET_ID
./ovlab dataset verify --dataset DATASET_ID
```

The built-in LIBERO bridge resolves the four OpenVLA RLDS suites
`libero_spatial`, `libero_object`, `libero_goal` and `libero_10` at a pinned
provider revision. URL imports require HTTPS and a SHA-256 digest. Embedded
credentials, unsafe redirects, traversal paths, archive links and excessive
extraction sizes are rejected. Local imports are copied into controlled storage
so an external mutable path never becomes a training input.

Dataset acquisition is never implicit. `train run --allow-dataset-download` is
the one explicit convenience that may acquire a known built-in selector before
training; without it a missing dataset is an actionable error.

## Profiles and planning

Portable YAML profiles live in `configs/experiments/training/` and use
`ovlab.training-profile/v1`. They contain immutable logical model and dataset
selectors, a seed, bounded hyperparameters, precision, training mode and output
semantics. Relative `extends` composes a shared base with variant overrides; mappings
merge and sequences replace. Absolute host paths, shell commands, unknown keys
and unresolved aliases are rejected.

```bash
./ovlab train profiles
./ovlab train validate --profile configs/experiments/training/openvla-libero10-lora-smoke.yaml
./ovlab train run openvla-libero10-quic-smoke.yaml
./ovlab train plan --profile openvla-libero10-quic-hard.yaml
./ovlab train plan --profile configs/experiments/training/openvla-libero10-lora-smoke.yaml
```

`model.revision` accepts a pinned 40-character commit, `local/main`, or
`remote/main`. Omitting it means `local/main`: both `train plan` and `train run`
read the cached Hugging Face `refs/main` and require that snapshot locally,
without a network request. `remote/main` queries Hugging Face for the current
commit; `train run` downloads its snapshot when needed, while `train plan`
requires it to be cached already. The selected commit and observed file hashes
are recorded in the training plan. If the registry also pins a commit, an
explicit commit must match it and its registered hashes are enforced. With a
moving registry entry, an explicit commit selects that cached snapshot and
records its observed file hashes.

`openvla-libero10-quic-hard.yaml` is the longer LIBERO-10 QuiC run: up to 20
episodes per task, an 80/20 episode split, one frame in eight, 2,000 optimizer
steps, effective batch size four, and up to 1,400 teacher-forced validation
batches. It inherits the smoke profile's frozen host action normalization;
pass `--fit-action-normalization` to fit train-only q01/q99 statistics instead.

Validation does not resolve large assets. Planning verifies local dataset and
base-checkpoint bytes, negotiates capabilities and evaluates resources, but
does not initialize OpenVLA, allocate CUDA or download anything. The generic
schema represents `full` and `peft`, including unquantized LoRA and the
experimental `quic` method. QuIC selects the `openvla-quic` trainer and the
independent `quic-petf-vla` method dependency. QLoRA and quantized training
remain unsupported.

The OpenVLA reference LoRA recipe uses `target_modules: [all-linear]`, an
explicit rank, `alpha: min(rank, 16)`, BF16 or FP32 and an unmerged adapter
output. Reference guidance estimates at least 27 GiB for unquantized LoRA;
planning fails if either the profile limit or detected GPU memory is lower.

## QuIC profiles and ablations

The QuIC smoke profile uses a dense, suite-finetuned LIBERO-10 host with its
inherited action statistics, explicit deterministic episode train/validation
partition, and adapter-only output:

```bash
./ovlab train validate --profile configs/experiments/training/openvla-libero10-quic-smoke.yaml
./ovlab train plan --profile configs/experiments/training/openvla-libero10-quic-smoke.yaml
./ovlab train run configs/experiments/training/openvla-libero10-quic-smoke.yaml
./ovlab train ablate validate --suite configs/ablations/openvla-quic-placement-smoke.yaml
./ovlab train ablate run --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-run
```

Suites validate declared factors and shared host/data/training invariants,
materialize ordinary resolved profiles, execute through the existing isolated
owner and compare immutable split/normalization manifests across child runs.
See [`docs/ABLATIONS.md`](docs/ABLATIONS.md) for plan/materialize commands, data
selection, source-workflow differences, and the vanilla normalization sidecar
limitation. A three-step smoke verifies wiring, not training quality.

Every explicit partition splits immutable TFDS episode IDs before frame
sampling. Training and validation cannot share an episode. QuIC defaults to frozen selected
host action statistics. `--fit-action-normalization` on `train validate|plan|run`
or any `train ablate` operation selects training-only q01/q99 fitting before
planning. It requires a partition, applies consistently to every suite variant,
changes the scientific identity, and leaves source profiles unchanged. The
resolved YAML alternative is `dataset.partition.action_normalization: train-q99`;
reference full/LoRA profiles also support this mode. Fitted statistics exclude
validation episodes, stay frozen throughout the run, and accompany the QuIC
adapter through reload and inference.
Teacher-forced validation remains separate from benchmark rollout evaluation.

```bash
./ovlab train run configs/experiments/training/openvla-libero10-quic-smoke.yaml \
  --fit-action-normalization
./ovlab train ablate run --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --fit-action-normalization --output /tmp/quic-placement-fitted-run
```

Training optimizer steps, QuIC teacher-forced validation, and rollout task/episode
progress write to stderr. In an interactive `train run`, Docker receives a
terminal and each progress bar updates in place. Worker logs retain periodic
plain-text snapshots. Redirected output and `--json` use plain progress lines
on stderr, leaving JSON stdout intact. `OVLAB_PROGRESS=off` disables the
console tracker; set the same variable on the relevant container/service
environment when invoking a worker directly. The short human result includes
separate checkpoint and validation checks: a completed smoke run can have a
valid checkpoint while teacher-forced action-token accuracy remains low.

## Isolated execution

Build the purpose-specific images after changing packaged source or locks:

```bash
bash deploy/scripts/build-images.sh dataset training-openvla training-openvla-quic reporting
```

Then start a bounded run:

```bash
./ovlab train run \
  configs/experiments/training/openvla-libero10-lora-smoke.yaml
```

The trainer has one requested GPU, no network, a read-only prepared dataset, a
read-only base checkpoint and one writable training-run directory. It cannot
write the finalized checkpoint registry. A second non-GPU, network-disabled
finalizer reopens and validates the staged safetensors, creates the manifest
last, and atomically publishes the bundle. The isolated reporting image then
reads the finalized training run read-only and publishes its performance
report. Base weights are never modified and LoRA weights remain unmerged.

Before starting the trainer, OVLAB's checkpoint resolver prepares a verified
managed snapshot with hardlinks, so Hugging Face blob symlinks cannot become
broken references inside the isolated mount. Planning stays read-only. The
mounted run keeps its canonical run ID in both the trainer and finalizer.
Run directories explicitly use shared-group permissions (`2770`, with `660`
for append-only metrics/events), independently of the host umask. Workers
receive the actual groups of their writable mounts; running as root or making
the run directory world-writable is unnecessary.
When finalization crosses bind mounts, the finalizer copies staged files after
an `EXDEV` link failure and publishes them through the same verified, atomic
checkpoint path. Worker-owned evidence receives shared-group read access,
including private Safetensors staging files, so host-side verification works.

Inspect canonical evidence offline:

```bash
./ovlab train status TRAINING_RUN_ID
./ovlab train inspect TRAINING_RUN_ID
./ovlab train verify TRAINING_RUN_ID
./ovlab train report --run TRAINING_RUN_ID
./ovlab train report --run TRAINING_RUN_ID --verify
./ovlab checkpoint list
./ovlab checkpoint inspect --checkpoint CHECKPOINT_ID
./ovlab checkpoint verify --checkpoint CHECKPOINT_ID
```

Add `--json` to any of these commands for one stable machine-readable document.
Checkpoint verification parses safetensors metadata and finite floating-point
payloads without importing or executing model code.

Each optimizer step records a versioned system-performance sample: PyTorch CUDA
allocator `allocated`, `reserved`, peak-allocated and peak-reserved bytes;
exact total/trainable/frozen/adapter parameter counts; and an explicitly
qualified estimated-GFLOPs value. VRAM is process allocator telemetry, not
whole-device NVML usage. The compute value is the documented analytical proxy
`(2 * total parameters + 4 * trainable parameters) * non-padding tokens`, not
a measured hardware counter. This keeps LoRA backward cost separate from the
frozen base-model forward path.

`train report` runs in the locked reporting image. It mounts
`runs/train/` read-only and writes a checksummed, self-contained HTML/JSON
bundle to:

```text
exports/derived/training/<training-run-id>/system-performance/<build-id>/
```

The report includes interactive VRAM and estimated-compute time series,
descriptive statistics, runtime parameter classes, lifecycle peaks and source
checksums. Use `--verify --build BUILD_ID` to verify an exact build. Report
generation never changes canonical training evidence or checkpoint identity.

## Deployment handoff

Deployment must select the content-derived `checkpoint-<32 hex>` ID explicitly.
The resolver rejects `latest`, incomplete/failed/interrupted staging output,
checksum failures, incompatible base dependencies and unsupported schemas. For
LoRA it resolves and records both the immutable base checkpoint and immutable
adapter bundle, mounts both read-only, and starts neither training nor dataset
access. A later portable experiment may expose this as:

```yaml
policy:
  provider: openvla
  checkpoint:
    id: checkpoint-0123456789abcdef0123456789abcdef
```

## Interruption and recovery

SIGINT or SIGTERM reaches the foreground trainer. Completed events and metrics
remain readable, the run becomes `interrupted`, partial checkpoints stay in
`staging-checkpoints/`, and no registry checkpoint is published. Retry dataset
acquisition explicitly after inspecting its `.staging` evidence. OVLAB removes
only containers it owns; it never uses global Docker prune operations.
