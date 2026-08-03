# Training operations

Detailed dataset-provider, acquisition, local-import and integrity procedures
are documented separately in [`DATASETS.md`](DATASETS.md). This guide focuses
on the end-to-end training and checkpoint lifecycle.

OVLAB manages training as a separate, reproducible pipeline:

```text
explicit dataset acquisition -> verified preparation -> validated profile
-> offline trainer -> finalized immutable checkpoint -> explicit deployment handoff
```

Training never runs as part of `deploy run`. Benchmark `runs/`, reporting
`derived/` and publication `exports/` remain separate from model data. The host
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
├── training-runs/<training-run-id>/
│   ├── original-profile.yaml
│   ├── resolved-profile.yaml
│   ├── training-plan.json
│   ├── provenance.json
│   ├── events.jsonl
│   ├── metrics.jsonl
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
Host paths and timestamps are provenance only; they do not affect scientific
identities. Failed or interrupted work remains evidence under staging or the
training run and is never advertised as ready.

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

Portable YAML profiles live in `configs/training/` and use
`ovlab.training-profile/v1`. They contain immutable logical model and dataset
selectors, a seed, bounded hyperparameters, precision, training mode and output
semantics. Absolute host paths, shell commands, unknown keys and unresolved
aliases are rejected.

```bash
./ovlab train profiles
./ovlab train validate --profile configs/training/openvla-libero10-lora-smoke.yaml
./ovlab train plan --profile configs/training/openvla-libero10-lora-smoke.yaml
```

Validation does not resolve large assets. Planning verifies local dataset and
base-checkpoint bytes, negotiates capabilities and evaluates resources, but
does not initialize OpenVLA, allocate CUDA or download anything. The generic
schema represents `full` and `peft`; Gate I registers unquantized LoRA as its
only PEFT method. QuIC, QLoRA and quantized training are deliberately rejected.

The OpenVLA reference LoRA recipe uses `target_modules: [all-linear]`, an
explicit rank, `alpha: min(rank, 16)`, BF16 or FP32 and an unmerged adapter
output. Reference guidance estimates at least 27 GiB for unquantized LoRA;
planning fails if either the profile limit or detected GPU memory is lower.

## Isolated execution

Build the purpose-specific images after changing packaged source or locks:

```bash
bash deploy/scripts/build-images.sh dataset training-openvla reporting
```

Then start a bounded run:

```bash
./ovlab train run \
  --profile configs/training/openvla-libero10-lora-smoke.yaml
```

The trainer has one requested GPU, no network, a read-only prepared dataset, a
read-only base checkpoint and one writable training-run directory. It cannot
write the finalized checkpoint registry. A second non-GPU, network-disabled
finalizer reopens and validates the staged safetensors, creates the manifest
last, and atomically publishes the bundle. The isolated reporting image then
reads the finalized training run read-only and publishes its performance
report. Base weights are never modified and LoRA weights remain unmerged.

Inspect canonical evidence offline:

```bash
./ovlab train status --run TRAINING_RUN_ID
./ovlab train inspect --run TRAINING_RUN_ID
./ovlab train verify --run TRAINING_RUN_ID
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
`training-runs/` read-only and writes a checksummed, self-contained HTML/JSON
bundle to:

```text
derived/training/<training-run-id>/system-performance/<build-id>/
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
