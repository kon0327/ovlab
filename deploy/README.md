# Reproducible OVLAB deployment

This directory packages the accepted OVLAB runtime boundaries without changing
the host Conda environments. Production containers expose only the `ovlab` CLI.
They communicate through a versioned AF_UNIX socket in `/run/ovlab`; no TCP
port, host networking, Docker socket, privileged mode, or source-tree bind is
used.

For a command-oriented build and launch guide, see
[`docker/README.md`](docker/README.md).

## Production image matrix

| Image | Runtime responsibility | Included heavy closure | Deliberately absent |
|---|---|---|---|
| `ovlab-benchmark-libero` | CLI, runner, LIBERO, Robosuite, MuJoCo, EGL and immutable trace output | LIBERO simulation stack; Torch only because pinned LIBERO imports it to load initial states | OpenVLA, Transformers, PEFT, FlashAttention and model weights |
| `ovlab-reporting` | Offline HTML reporting plus isolated/grouped publication exports | NumPy, Jinja2 and Matplotlib | LIBERO, MuJoCo, model runtimes, checkpoints, datasets, GPU and policy socket |
| `ovlab-dataset` | Explicit dataset acquisition, offline verification and checkpoint finalization | dependency-light CLI plus pinned archive support | model runtimes, LIBERO, benchmark runs and GPU |
| `ovlab-training-openvla` | Offline full/LoRA OpenVLA training | Torch/CUDA, Transformers, PEFT, FlashAttention, TensorFlow/RLDS and pinned OpenVLA source | LIBERO runtime, benchmark runs, reporting and finalized-checkpoint write access |
| `ovlab-policy-openvla` | Vanilla and merged-LoRA OpenVLA service | Torch 2.2.0/CUDA 12.1, Transformers 4.40.1, PEFT 0.11.1, BitsAndBytes 0.43.1, FlashAttention 2.5.5 and pinned OpenVLA source | LIBERO, Robosuite, MuJoCo, datasets and run output |
| `ovlab-policy-openvla-oft` | OpenVLA-OFT service | The distinct OFT Transformers commit, Torch/CUDA, BitsAndBytes 0.43.1, FlashAttention and pinned OFT source | LIBERO, Robosuite, MuJoCo, datasets and run output |

Vanilla, merged-LoRA and OFT policies accept
`runtime.quantization: none | 8bit | 4bit`. The `8bit` identity denotes the
BitsAndBytes LLM.int8 runtime, while `4bit` denotes the pinned NF4 recipe with
BF16 compute, FP16 storage and double quantization. Applying either mode to a
published fine-tuned artifact is inference-time quantization; it is not
evidence that the model or adapter was trained with QLoRA.

QuIC remains descriptor-only. There is no QuIC image, Compose profile, provider
selection, or claim of runtime readiness in Gate H.

## Reproducibility and provenance

Base images are pinned by digest in `docker/base-images.lock`. Python closures
are resolved into PEP 751 `locks/*.pylock.toml` files with artifact SHA-256
hashes. Docker consumes generated `--require-hashes` requirement files with
dependency resolution disabled. The OFT Transformers fork is the single VCS
artifact; both requested revision and resolved commit are
`bc339d9ad707454c0c115970db43c260067c61ab`.

`scripts/source_manifest.py` records HEAD, clean/dirty state, submodule gitlinks,
untracked build-relevant files, and content hashes deterministically. The build
script embeds that manifest and adds OCI labels for repository revision, source
hash, dirty state, lock hash, Dockerfile hash and image role. Dirty builds are
allowed but never presented as clean. Portable files below `configs/` are not
image inputs: their identity belongs to the per-run configuration bundle, so
adding or changing an experiment does not require rebuilding an image.

Build all mandatory production images:

```bash
bash deploy/scripts/build-images.sh
```

The host needs Docker Engine with Compose v2, NVIDIA Container Toolkit, a driver
compatible with CUDA 12.1, and enough local storage for the pinned images. Verify
these prerequisites with `docker version`, `docker compose version`, `docker info`,
`nvidia-smi`, and a small `docker run --gpus all` check.

For a deployment, set the four `OVLAB_*_IMAGE` Compose variables to immutable
registry digest references and set the matching `*_IMAGE_DIGEST` values. Those
identities enter execution provenance and the policy handshake, but never the
scientific configuration hash. Machine paths, socket names, renderer device IDs
and timing also remain outside the scientific hash.

Copy `compose/.env.example` to an untracked `.env` and set local paths. Record the
deployment identity before launch, for example:

```bash
export OVLAB_DEPLOYMENT_MANIFEST_SHA256="$(sha256sum deploy/compose/compose.yaml | cut -d' ' -f1)"
export OVLAB_CONTAINER_RUNTIME_VERSION="$(docker version --format '{{.Server.Version}}')"
```

## Runtime mounts and permissions

All containers run as UID/GID `10001:10001`, drop all capabilities, enable
`no-new-privileges`, use a read-only root filesystem and have no network. The
policy receives only its resolved snapshot at
`/checkpoints/resolved/<checkpoint-id>:ro` and the shared socket volume. The
benchmark receives `/datasets:ro`, the socket volume and the host-backed
`/var/lib/ovlab/runs` directory as its only persistent writable mount. Canonical
run evidence is therefore directly available outside the container. Both runtime
services receive the same minimal configuration bundle at `/opt/ovlab/configs:ro`.
The host CLI validates and materializes this bundle for one deployment and removes
the temporary host directory after project-scoped teardown.

The benchmark image also contains a portable LIBERO path map at
`/etc/ovlab/libero/config.yaml`. It points bundled BDDL, initial-state and asset
paths into the pinned LIBERO source and maps demonstrations to `/datasets`.
`LIBERO_CONFIG_PATH=/etc/ovlab/libero` prevents upstream LIBERO from trying to
create an interactive configuration below the non-root user's read-only home.

Configure host deployment locations when necessary:

```bash
export OVLAB_GLOBAL_HF_CACHE=/home/kony/.cache/huggingface
export OVLAB_MANAGED_CHECKPOINTS_ROOT=/home/kony/dissertation/ovlab-data/checkpoints/huggingface
# Optional override; otherwise derived as ovlab-data/datasets/libero.
export OVLAB_DATASETS_PATH=/home/kony/dissertation/ovlab-data/datasets/libero
export OVLAB_RUNS_ROOT=/home/kony/dissertation/ovlab-data/runs
export OVLAB_DERIVED_ROOT=/home/kony/dissertation/ovlab-data/derived
export OVLAB_EXPORTS_ROOT=/home/kony/dissertation/ovlab-data/exports
export OVLAB_GPU_DEVICE=0
export OVLAB_EGL_DEVICE_ID=0
```

These are deployment settings, not scientific parameters. Do not put secrets in
them. Hugging Face and Transformers offline modes are forced in policy services;
only the host orchestrator may download a missing pinned artifact before the
containers start.

Create the host artifact workspace before launching containers. It is deliberately
outside the source checkout:

```text
/home/kony/dissertation/ovlab-data/
├── checkpoints/
│   ├── huggingface/   # pinned snapshots managed or materialized by OVLAB
│   └── local/         # unpublished QuIC and experimental checkpoints
├── datasets/
│   └── <provider>/... # immutable raw and prepared training datasets
├── training-runs/     # canonical training evidence and staged output
├── runs/              # immutable traces, per-episode videos, metrics, config and provenance
├── derived/           # regenerated reports, plots and tables
└── exports/           # curated outputs prepared for papers and sharing
```

## Checkpoint resolution

Experiments contain only a logical `settings.checkpoint_id`. The portable
registry supplies its Hugging Face repository, pinned revision, aggregate hash,
and exact file identities. Before Compose starts, `ovlab deploy run` resolves
that identity in this order:

1. `${OVLAB_GLOBAL_HF_CACHE}`, defaulting to `~/.cache/huggingface`;
2. `resources.checkpoints.<id>.local_path` in an explicitly selected gitignored
   local profile;
3. `${OVLAB_MANAGED_CHECKPOINTS_ROOT}`, defaulting to
   `<OVLAB_RUNS_ROOT>/../checkpoints/huggingface`;
4. a host-side download of the pinned revision into managed storage.

Every selected artifact is checked against registry sizes and SHA-256 values
before Docker starts. A global Hugging Face snapshot is materialized into the
managed tree with hard links, so its weight bytes are not copied and its cache
symlinks remain valid after mounting only the selected snapshot. Cross-filesystem
materialization fails clearly instead of silently copying large weights.
During a managed download the CLI reports repository metadata, file number,
filename, byte progress, completion, and verification stages on stderr.

Use strict offline resolution when downloads are forbidden:

```bash
./ovlab deploy run EXPERIMENT --offline
```

For a custom checkpoint, pass a local profile explicitly:

```bash
./ovlab deploy run EXPERIMENT \
  --local-profile configs/local/profile.yaml
```

The policy always receives `/checkpoints/resolved/<checkpoint-id>`. The host
path and discovery source are execution provenance; checkpoint ID, repository,
revision, file identities, and hashes remain scientific identity.

`OVLAB_DATASETS_PATH` is optional. When unset, the CLI derives
`<OVLAB_RUNS_ROOT>/../datasets/libero`, creates the directory, and mounts it at
`/datasets:ro`. An explicit override must already exist, preventing a misspelled
path from being silently created. LIBERO task definitions, assets, and initial
states remain supplied by the pinned benchmark source; demonstration datasets
belong only in this host data tree.

The container runtime uses UID/GID `10001:10001`. The host CLI creates missing
`runs/`, `derived/`, and `exports/` roots with mode `2770`, creates the default `datasets/libero/` root when
absent, and passes the invoking user's primary GID as a
supplementary container group, so canonical artifacts remain writable by the
host user without world-writable permissions. `runs/` is writable only in benchmark services.
After benchmark/policy teardown, the reporting container mounts `runs/`
read-only and writes below `derived/` and `exports/`. Benchmark services never
mount either output root. Grouped exports remain an explicit operator workflow. Policy containers mount
none of these artifact roots. Host paths and Docker tags are deployment
provenance and never enter the scientific configuration hash.

Finalized training artifacts use content-derived `checkpoint-<32 hex>` IDs.
Their deployment handoff resolves the verified read-only adapter plus its
registry-pinned read-only base checkpoint. Mutable aliases, failed/interrupted
training output and staging checkpoints are not deployable. See
[`../TRAINING.md`](../TRAINING.md) for commands and schema details.

## CLI-managed Compose profiles

Normal operation uses the repository launcher. It requires Docker and system
Python 3, not an activated Conda environment:

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml
```

The CLI passes the selected experiment to both runtime services, performs checkpoint
resolution and Compose preflight, propagates the benchmark status and reaps the
project-scoped containers and RPC volume. It then hands the single completed run
ID to the isolated reporting service. It checks a versioned deployment
contract on all selected images before hashing a large checkpoint, so stale
images fail with an exact rebuild command instead of running old mount semantics.
New or changed YAML below `configs/` is resolved into a read-only per-run bundle
and does not require a rebuild. After changing packaged Python source, dependency
locks, Dockerfiles, or bundled external source, rebuild the selected topology:

```bash
bash deploy/scripts/build-images.sh benchmark reporting policy-openvla-oft
```

The commands below describe the underlying topology and remain useful for
diagnostics.

The accepted OpenVLA topology is:

```bash
docker compose --file deploy/compose/compose.yaml --profile openvla up \
  --abort-on-container-exit --exit-code-from benchmark-openvla
```

The separate OFT topology is selected with `--profile oft`. The benchmark waits
for a protocol-aware health probe. Readiness is advertised only after provider
initialization; the health operation performs neither prediction nor trace
creation and does not consume the service connection.

GPU access uses Compose `gpus: all` plus `NVIDIA_VISIBLE_DEVICES`; set a concrete
device in reproducible deployments. The benchmark forces EGL. GLFW remains an
interactive, dependency-light-tested playground path and is not required by
headless deployment.

For an interactive WSLg/X11 playground, layer the GLFW overlay on the primary
project:

```bash
docker compose \
  --file deploy/compose/compose.yaml \
  --file deploy/compose/compose.glfw.yaml \
  --profile openvla up --abort-on-container-exit \
  --exit-code-from benchmark-openvla
```

The overlay selects `profiles/libero-playground-glfw.yaml`, explicitly forces
`MUJOCO_GL=glfw`, removes `MUJOCO_EGL_DEVICE_ID`, and mounts only the X11 Unix
socket read-only. Set `OVLAB_DISPLAY` and `OVLAB_X11_SOCKET` when their host
values differ. GLFW still requires a live X11/Wayland display server; hiding a
window does not make it a headless backend. Use EGL for automated or unattended
execution. The production image intentionally does not bundle Xvfb.

The containers preserve the Gate G workflow through the same public entrypoint:
use `ovlab config validate ...` or `ovlab config resolve ...`, start the selected
foreground `ovlab service serve ...`, use `ovlab connect ...` for a no-prediction
capability probe, and then `ovlab run ...`. Add `--json` where supported for stable
machine output; omit it for human output. Compose delegates directly to the service
and run commands and returns their exit status.

Publish the report and isolated export without granting write access to
canonical evidence:

```bash
export OVLAB_REPORT_RUN_ID=<canonical-run-directory>
export OVLAB_REPORT_PROFILE=libero-task-default
docker compose --file deploy/compose/compose.yaml --profile reporting run --rm reporting
```

The source is mounted at `/var/lib/ovlab/runs:ro`; the report is written below
`/var/lib/ovlab/derived/<run>/<profile>/<derived-build-id>` and the isolated
export below `/var/lib/ovlab/exports/isolated/<run>`. Report generation performs
neither policy inference nor LIBERO execution. Only `runs/` is writable during
the benchmark; reporting/export failure is isolated from canonical finalization.

Generate a grouped export with canonical runs mounted read-only:

```bash
docker compose --file deploy/compose/compose.yaml --profile export run --rm export \
  export grouped --name paper-ablation --runs RUN_ID_A RUN_ID_B
```

The export service writes only below `/var/lib/ovlab/exports`. Both standalone
services retain the non-root, read-only-root, dropped-capability,
`no-new-privileges` and `network_mode: none` contract.

Use `docker compose ... down` for scoped teardown. Do not use global prune commands.
On a signal, the foreground CLI closes its adapter and removes only its owned socket;
partial run artifacts already created remain available for inspection.

## Test-only transport smoke

The mock provider exists only in `Dockerfile.smoke`; no production image copies
it. It validates two non-root containers, the shared socket, health ordering,
capability negotiation and cleanup without loading a model or LIBERO:

```bash
bash deploy/scripts/transport-smoke.sh
```

This smoke is evidence for packaging and IPC only. It is not scientific model or
benchmark evidence.

## Troubleshooting and limitations

- A missing checkpoint under `--offline` is an expected hard failure. Without
  `--offline`, inspect host resolver diagnostics, registry revision, network
  access, and managed-storage permissions. Policy runtime networking remains
  disabled.
- For `cuda is not available`, verify the NVIDIA runtime, driver compatibility and
  selected `OVLAB_GPU_DEVICE`.
- For EGL failures, validate `MUJOCO_EGL_DEVICE_ID`, graphics driver exposure and the
  detected renderer. WSL may expose Mesa `llvmpipe`; this is valid headless EGL but
  is software rendering, not NVIDIA renderer evidence.
- Socket permission failures usually mean the shared volume was not created by the
  Compose project or the fixed UID/GID contract was overridden inconsistently.
- The default container UID/GID is `10001:10001`; ensure the writable runs directory
  accepts that identity without making it world-writable.

Gate H validates packaging, isolation, IPC, CUDA visibility and headless startup. It
does not validate model output, task success, latency, full evaluation, QuIC math or
deployment safety. QuIC-PEFT and QuIC-WC remain descriptor-only and non-runnable.
