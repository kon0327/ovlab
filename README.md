# OpenVLABenchmark (OVLAB)

OpenVLABenchmark (OVLAB) is a reproducible experimental framework for evaluating OpenVLA-derived Vision–Language–Action policies, initially with the LIBERO benchmark.

Current project version: **0.2.0**. See the complete
[`release notes`](RELEASE_NOTES.md) for implemented capabilities, compatibility
boundaries and current scope limits.

The complete system design, runtime boundaries, data flows and extension points
are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

OVLAB follows a **Config → Connect → Run → Inspect** workflow: define an experiment, connect interchangeable policies and benchmarks through shared contracts, execute reproducible runs, and validate or analyze their immutable traces offline.

The experiment runner and benchmark adapter execute together in a runner process or container. Each VLA implementation runs independently as a policy service, communicating through a dependency-light policy protocol. This separation allows Vanilla, LoRA, OFT, and QuIC implementations to keep distinct dependency environments.

## Repository layout

- `code/`: core packages, the policy SDK, benchmark adapters, metrics, policy integrations, applications, and tests.
- `configs/`: benchmark, policy, metric, protocol, and experiment configuration.
- `deploy/`: reserved for Docker, Compose, and deployment scripts.
- `external/`: destinations for pinned external repositories and the dedicated OpenVLA-QuIC fork.
- `runs/`: optional repo-local development output; contents are not versioned.

Container deployments use a host-backed artifact workspace outside the source
checkout. The canonical convention is `ovlab-data/runs` for immutable evidence,
`ovlab-data/checkpoints` for orchestrator-managed model artifacts,
`ovlab-data/datasets` for benchmark datasets,
`ovlab-data/derived` for regenerated analyses, and `ovlab-data/exports` for
curated publication outputs. Existing snapshots in the user's global Hugging
Face cache are reused through verified hard links rather than copied. Only
benchmark containers write canonical runs; reporting mounts them read-only.
See [`deploy/README.md`](deploy/README.md).

Dataset discovery, explicit acquisition, local import, verification and storage
are documented in the [`Dataset operations guide`](DATASETS.md). Versioned
training profiles, isolated OpenVLA-LoRA training and immutable checkpoint
handoff are documented in the [`Training operations guide`](TRAINING.md).

## Command line

Use the unified CLI directly from a checkout without installing packages:

```bash
./ovlab --help
./ovlab deploy run EXPERIMENT --profile openvla --renderer egl
./ovlab config validate CONFIG --mode descriptor
./ovlab policy list
./ovlab connect CONFIG
./ovlab run CONFIG --dry-run
./ovlab run inspect RUN_PATH
./ovlab run verify RUN_PATH
./ovlab metrics recompute RUN_PATH
./ovlab report publish --run RUN_ID --profile libero-task-default
./ovlab report generate --run RUN_ID --profile libero-task-default
./ovlab report verify --run RUN_ID --profile libero-task-default --build BUILD_ID
./ovlab export isolated --run RUN_ID
./ovlab export grouped --name STUDY_NAME --runs RUN_ID_A RUN_ID_B
./ovlab dataset resolve --benchmark libero --suite libero_10
./ovlab train validate --profile configs/training/openvla-libero10-lora-smoke.yaml
./ovlab train report --run TRAINING_RUN_ID
./ovlab checkpoint list
```

`ovlab deploy run` is the normal operator workflow. It launches the selected
policy service and the LIBERO runner as separate Compose containers, waits for
readiness, propagates the benchmark result, and cleans up the private deployment
resources. It then starts the dedicated reporting image with canonical `runs/`
mounted read-only and publishes `derived/` plus the isolated `exports/`. It needs
Docker and system Python 3, not an activated Conda environment.

Set `OVLAB_LOCAL_PROFILE` to a gitignored machine profile only for low-level
native development. Installing `ovlab-benchctl` exposes the same command as the
`ovlab` console entrypoint. Complete command, output, exit-code, foreground
service, cleanup, and QuIC-skeleton behavior is documented in the
[`OVLAB CLI guide`](code/apps/benchctl/CLI_README.md).

Report generation, offline viewing, integrity verification, isolated/grouped exports,
and the production reporting-container workflow are covered in the
[`Reporting and export operations guide`](REPORTING.md).

## Testing

OVLAB uses the lightweight `ovlab-tester` Conda environment for CPU-only automated tests. Create it from `deploy/environments/ovlab-tester/environment.yml`, then run:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh
```

GPU and policy-specific smoke tests remain isolated in their corresponding policy environments.
