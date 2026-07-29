# OVLAB Docker deployment

This directory contains the production Dockerfiles for the isolated OVLAB
runtime. The public `ovlab deploy run` command owns the Compose lifecycle; users
do not need to activate Conda or start containers separately. The Dockerfiles
are not intended to collapse the benchmark and policy service into one
container.

For the complete reproducibility and security contract, see
[`deploy/README.md`](../README.md). Dependency locks are documented in
[`deploy/locks/README.md`](../locks/README.md).

## Runtime topology

| Image | Responsibility | Persistent access |
|---|---|---|
| `ovlab-benchmark-libero:local` | OVLAB CLI, runner, LIBERO, Robosuite and MuJoCo | datasets read-only, canonical runs read-write |
| `ovlab-policy-openvla:local` | OpenVLA Vanilla and merged-LoRA policy service | checkpoints read-only |
| `ovlab-policy-openvla-oft:local` | OpenVLA-OFT policy service | checkpoints read-only |

The benchmark and policy containers communicate only through a versioned
AF_UNIX socket in a Compose-managed volume. They expose no TCP service and run
with networking disabled. Model weights and datasets are never copied into an
image.

## Prerequisites

- Docker Engine with Compose v2;
- NVIDIA Container Toolkit and a CUDA 12.1-compatible host driver;
- local, complete checkpoint and LIBERO dataset trees;
- sufficient storage for the three pinned images;
- an EGL-capable device for unattended benchmark execution.

Check the host before building:

```bash
docker version
docker compose version
docker info
nvidia-smi
```

## Build images

Run builds from the repository root:

```bash
bash deploy/scripts/build-images.sh
```

Build only one role when iterating:

```bash
bash deploy/scripts/build-images.sh benchmark
bash deploy/scripts/build-images.sh policy-openvla
bash deploy/scripts/build-images.sh policy-openvla-oft
```

The build uses pinned base-image digests and hash-locked Python artifacts. It
embeds a source manifest and marks an image as dirty when the source worktree is
dirty. Building does not modify the host Conda environments.

Inspect the resulting identities:

```bash
docker image inspect ovlab-benchmark-libero:local
docker image inspect ovlab-policy-openvla:local
docker image inspect ovlab-policy-openvla-oft:local
```

## Prepare host data

Keep deployment evidence outside the source checkout:

```text
/home/kony/dissertation/ovlab-data/
├── checkpoints/
│   ├── huggingface/   # pinned snapshots managed by the OVLAB CLI
│   └── local/         # unpublished QuIC and experimental artifacts
├── datasets/
│   └── libero/        # LIBERO demonstrations mounted read-only
├── runs/              # canonical traces, metrics, video, config and provenance
├── derived/           # regenerated reports, plots and tables
└── exports/           # curated publication and sharing outputs
```

The CLI creates a missing `runs/` directory with mode `2770` and passes the
invoking user's primary GID to benchmark containers as a supplementary group.
For direct Compose use, create the directories yourself and set
`OVLAB_HOST_ARTIFACT_GID` to `id -g`. Only the benchmark writes canonical runs.
Reporting mounts `runs/` read-only, and `exports/` is not mounted into OVLAB
containers.

Copy the environment template and replace every host-specific value:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

At minimum, configure:

```dotenv
OVLAB_GLOBAL_HF_CACHE=/home/kony/.cache/huggingface
OVLAB_MANAGED_CHECKPOINTS_ROOT=/home/kony/dissertation/ovlab-data/checkpoints/huggingface
# Optional; defaults to /home/kony/dissertation/ovlab-data/datasets/libero
OVLAB_DATASETS_PATH=/home/kony/dissertation/ovlab-data/datasets/libero
OVLAB_RUNS_ROOT=/home/kony/dissertation/ovlab-data/runs
OVLAB_DERIVED_ROOT=/home/kony/dissertation/ovlab-data/derived
OVLAB_EXPORTS_ROOT=/home/kony/dissertation/ovlab-data/exports
OVLAB_GPU_DEVICE=0
OVLAB_EGL_DEVICE_ID=0
```

Host paths, renderer device IDs, socket names and Docker tags are execution
settings. They do not change the scientific configuration hash.

The checkpoint variables are discovery and storage roots, not direct policy
mounts. The CLI resolves the experiment's logical checkpoint ID, verifies its
pinned registry revision and file hashes, and mounts only that snapshot at
`/checkpoints/resolved/<checkpoint-id>:ro`. Existing global Hugging Face cache
data is reused with hard links instead of copying weight bytes.
When `OVLAB_DATASETS_PATH` is omitted, the CLI derives `datasets/libero` next
to `runs`, creates it if necessary, and mounts it read-only into the benchmark.

## Validate Compose

Preview the resolved orchestration without starting containers:

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
  --profile openvla \
  --renderer egl \
  --dry-run
```

The real command performs its own `docker compose config --quiet` preflight. It
also verifies a versioned deployment-contract label on both selected images, so
a stale image fails with an exact `build-images.sh` command before checkpoint
verification or container startup. After changing packaged source, rebuild the
selected topology, for example:

```bash
bash deploy/scripts/build-images.sh benchmark policy-openvla-oft
```

## Run with EGL

EGL is the supported backend for headless and automated LIBERO execution:

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
  --profile openvla \
  --renderer egl
```

Run OpenVLA-OFT separately:

```bash
./ovlab deploy run \
  configs/experiments/libero10-openvla-oft-rpc-smoke.yaml \
  --profile oft \
  --renderer egl
```

The policy service loads checkpoints offline, becomes healthy only after
initialization and is then consumed by the benchmark through the local socket.
Canonical output is written below `${OVLAB_RUNS_ROOT}` on the host.

By default, a checkpoint absent from the global and managed caches is downloaded
at its pinned revision by the host orchestrator before policy startup. For a
strictly offline run, require an already verified artifact:

```bash
./ovlab deploy run \
  configs/experiments/libero10-openvla-oft-rpc-smoke.yaml \
  --profile oft \
  --renderer egl \
  --offline
```

## Run the interactive GLFW path

GLFW requires a live X11 or Wayland display such as WSLg. It is not a truly
headless backend; hiding a window does not remove the display-server dependency.

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
  --profile openvla \
  --renderer glfw
```

The overlay forces `MUJOCO_GL=glfw`, removes the EGL device variable and mounts
the host X11 socket read-only. Use EGL for CI and unattended benchmarks.

## Generate a report

Set a canonical run directory name and a new output directory name:

```bash
export OVLAB_REPORT_RUN_ID=<run-directory-name>
export OVLAB_REPORT_OUTPUT_NAME=<new-derived-directory-name>

docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml \
  --profile reporting run --rm reporting
```

The report service reads `/var/lib/ovlab/runs` read-only and writes only to
`/var/lib/ovlab/derived`. The requested output directory must not already exist.

## Shutdown and verification

`ovlab deploy run` always executes project-scoped Compose teardown, including
its private RPC volume, on success, failure or interruption. Canonical bind-
mounted run artifacts are not removed.

Do not use a global Docker prune as part of the OVLAB workflow. After a run,
verify that no service remains alive and inspect the host artifacts directly:

```bash
docker ps --filter name=ovlab
find "${OVLAB_RUNS_ROOT}" -maxdepth 2 -type f
```

## Troubleshooting

- `permission denied` below the runs directory: correct ownership or ACLs for
  container UID/GID `10001:10001`.
- checkpoint not found under `--offline`: verify the logical ID, pinned registry
  revision, global cache, managed cache, or local-profile override. Without
  `--offline`, inspect host network and managed-storage permissions. Policy
  containers never receive network access.
- CUDA unavailable: check the NVIDIA Container Toolkit, host driver and
  `OVLAB_GPU_DEVICE`.
- EGL initialization failure: verify graphics-device exposure and
  `OVLAB_EGL_DEVICE_ID`; use GLFW only when a working display server is present.
- policy health timeout: inspect the policy container logs for offline checkpoint,
  CUDA, FlashAttention or compatibility errors.
- socket permission failure: do not override the fixed UID/GID independently
  between benchmark and policy services.
