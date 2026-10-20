# OpenVLABenchmark (OVLAB)

OpenVLABenchmark (OVLAB) is a reproducible experimental framework for evaluating OpenVLA-derived Vision–Language–Action policies, initially with the LIBERO benchmark.

OVLAB is authored by **Jiří Konečný**, VSB – Technical University of Ostrava.

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
- `runs/evaluate/` and `runs/train/`: optional local evaluation and training
  output; contents are not versioned.

Container deployments use a host-backed artifact workspace outside the source
checkout. The canonical convention is `ovlab-data/runs` for immutable evidence,
`ovlab-data/checkpoints` for orchestrator-managed model artifacts,
`ovlab-data/datasets` for benchmark datasets, and `ovlab-data/exports` for
regenerated reports, dashboards, and curated publication outputs. Existing snapshots in the user's global Hugging
Face cache are reused through verified hard links rather than copied. Only
benchmark containers write canonical runs; reporting mounts them read-only.
See [`deploy/README.md`](deploy/README.md).

Dataset discovery, explicit acquisition, local import, verification and storage
are documented in the [`Dataset operations guide`](DATASETS.md). Versioned
training profiles, isolated OpenVLA-LoRA training and immutable checkpoint
handoff are documented in the [`Training operations guide`](TRAINING.md).
Evaluation invocation and the difference between Docker deployment and a
machine-local run are described in [`RUN.md`](RUN.md).
Safe listing, archival and deletion of benchmark runs, derived reports, and
publication exports are documented in the [`Data management guide`](DATA_MANAGEMENT.md).

## Command line

Use the unified CLI directly from a checkout without installing packages:

```bash
./ovlab --help
./ovlab doctor
./ovlab doctor --experiment configs/experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml
./ovlab eval run libero10-lora-merged-rpc-smoke
./ovlab eval run libero10-lora-merged-rpc-smoke --quantization 4bit
./ovlab train run openvla-libero10-lora-smoke
./ovlab train run openvla-libero10-lora-smoke --dry-run
./ovlab deploy run openvla-libero10 --dry-run
./ovlab deploy run openvla-libero10
./ovlab deploy inspect openvla-libero10
./ovlab deploy verify openvla-libero10
./ovlab config validate CONFIG --mode descriptor
./ovlab policy list
./ovlab connect CONFIG
./ovlab run CONFIG --dry-run
./ovlab run EXPERIMENT --dry-run
./ovlab run EXPERIMENT --profile configs/local/profile.yaml --dry-run
./ovlab eval inspect RUN_PATH_OR_ID_OR_HASH
./ovlab eval verify RUN_PATH_OR_ID_OR_HASH
./ovlab metrics recompute RUN_PATH_OR_ID_OR_HASH
./ovlab report publish --run RUN_ID_OR_HASH --profile libero-task-default
./ovlab report generate --run RUN_ID_OR_HASH --profile libero-task-default
./ovlab report verify --run RUN_ID_OR_HASH --profile libero-task-default --build BUILD_ID
./ovlab export isolated --run RUN_ID_OR_HASH
./ovlab export grouped --name STUDY_NAME --runs RUN_ID_OR_HASH_A RUN_ID_OR_HASH_B
./ovlab data list
./ovlab data archive --run RUN_ID
./ovlab dataset resolve --benchmark libero --suite libero_10
./ovlab train validate --profile configs/experiments/training/openvla-libero10-lora-smoke.yaml
./ovlab train report --run TRAINING_RUN_ID
./ovlab checkpoint list
```

`./ovlab data list --kind runs` prints each canonical run ID followed by its
eight-character lookup hash in parentheses. Every benchmark-run selector in
the CLI accepts either the complete ID or that hash; training run selectors
use the same rule within their separate namespace. Exact IDs take precedence,
and an ambiguous hash is rejected.

`ovlab eval run` is the normal evaluation workflow. It launches the selected
policy service and the LIBERO runner as separate Compose containers, waits for
readiness, propagates the benchmark result, and cleans up the private deployment
resources. It then starts the dedicated reporting image with canonical `runs/`
mounted read-only and publishes all generated outputs below `exports/`. It needs
Docker and system Python 3, not an activated Conda environment.

`train run`, `eval run`, and `deploy run` select YAML by name or path from
`configs/experiments/training`, `configs/experiments/evaluation`, and `configs/deploy`.
Each supports `--dry-run`; `inspect TARGET` and `verify TARGET` inspect their runs
or services. `deploy run` builds and retains only the model service, with optional
RPC ports and quantization; see [model deployment](DEPLOYMENT.md).

Set `OVLAB_LOCAL_PROFILE` to a gitignored machine profile when resolving a
machine-local checkpoint path. Installing `ovlab-benchctl` exposes the same command as the
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

The loose checkpoint revision paths have CPU-only train-plan and eval-config
fixtures. They cover an omitted revision and explicit `local/main`, downloaded
`remote/main`, remote planning without downloads, and a registered pinned
commit. Run just that matrix with:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/unit/training/test_revision_scenarios.py \
  code/tests/unit/cli/test_checkpointing.py \
  code/tests/unit/cli/test_revision_deployment_scenarios.py
```

GPU and policy-specific smoke tests remain isolated in their corresponding policy environments.

The experimental QuIC integration, portable method, isolated images and
supported migration scope are described in [QuIC migration](docs/QUIC_MIGRATION.md).
Shared training variants use the [ablation workflow](docs/ABLATIONS.md).
