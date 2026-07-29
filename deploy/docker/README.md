# OVLAB Docker deployment

This directory contains the production Dockerfiles for the isolated OVLAB
runtime. Use the Compose definitions in `deploy/compose/` to run them; the
Dockerfiles are not intended to collapse the benchmark and policy service into
one container.

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
├── runs/       # canonical traces, metrics, video, config and provenance
├── derived/    # regenerated reports, plots and tables
└── exports/    # curated publication and sharing outputs
```

Create all three directories before starting Compose. Benchmark and reporting
containers run as UID/GID `10001:10001`; grant that identity write access to
`runs/` and `derived/` without making them world-writable. Only the benchmark
writes canonical runs. Reporting mounts `runs/` read-only, and `exports/` is not
mounted into OVLAB containers.

Copy the environment template and replace every host-specific value:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

At minimum, configure:

```dotenv
OVLAB_CHECKPOINTS_PATH=/absolute/path/to/huggingface
OVLAB_DATASETS_PATH=/absolute/path/to/libero
OVLAB_RUNS_ROOT=/home/kony/dissertation/ovlab-data/runs
OVLAB_DERIVED_ROOT=/home/kony/dissertation/ovlab-data/derived
OVLAB_EXPORTS_ROOT=/home/kony/dissertation/ovlab-data/exports
OVLAB_GPU_DEVICE=0
OVLAB_EGL_DEVICE_ID=0
```

Host paths, renderer device IDs, socket names and Docker tags are execution
settings. They do not change the scientific configuration hash.

## Validate Compose

Resolve the configuration without starting containers:

```bash
docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml \
  --profile openvla config --quiet
```

For the OFT topology, replace `openvla` with `oft`.

## Run with EGL

EGL is the supported backend for headless and automated LIBERO execution:

```bash
docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml \
  --profile openvla up \
  --abort-on-container-exit \
  --exit-code-from benchmark-openvla
```

Run OpenVLA-OFT separately:

```bash
docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml \
  --profile oft up \
  --abort-on-container-exit \
  --exit-code-from benchmark-oft
```

The policy service loads checkpoints offline, becomes healthy only after
initialization and is then consumed by the benchmark through the local socket.
Canonical output is written below `${OVLAB_RUNS_ROOT}` on the host.

## Run the interactive GLFW path

GLFW requires a live X11 or Wayland display such as WSLg. It is not a truly
headless backend; hiding a window does not remove the display-server dependency.

```bash
docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml \
  --file deploy/compose/compose.glfw.yaml \
  --profile openvla up \
  --abort-on-container-exit \
  --exit-code-from benchmark-openvla
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

Stop only the selected Compose project:

```bash
docker compose \
  --env-file deploy/compose/.env \
  --file deploy/compose/compose.yaml down
```

Do not use a global Docker prune as part of the OVLAB workflow. After a run,
verify that no service remains alive and inspect the host artifacts directly:

```bash
docker ps --filter name=ovlab
find "${OVLAB_RUNS_ROOT}" -maxdepth 2 -type f
```

## Troubleshooting

- `permission denied` below the runs directory: correct ownership or ACLs for
  container UID/GID `10001:10001`.
- checkpoint or dataset not found: verify the absolute bind-mount source; runtime
  networking is disabled and cannot download missing artifacts.
- CUDA unavailable: check the NVIDIA Container Toolkit, host driver and
  `OVLAB_GPU_DEVICE`.
- EGL initialization failure: verify graphics-device exposure and
  `OVLAB_EGL_DEVICE_ID`; use GLFW only when a working display server is present.
- policy health timeout: inspect the policy container logs for offline checkpoint,
  CUDA, FlashAttention or compatibility errors.
- socket permission failure: do not override the fixed UID/GID independently
  between benchmark and policy services.

