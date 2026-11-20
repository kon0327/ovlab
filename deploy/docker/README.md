# Runtime image recipes

This directory owns the benchmark, reporting and dataset infrastructure images.
Model runtime recipes live with their independently registered plugin projects:

| Project | Recipes |
| --- | --- |
| [`openvla`](../../plugins/openvla/README.md) | Vanilla/LoRA policy, reference training, checkpoint finalizer. |
| [`openvla-oft`](../../plugins/openvla-oft/README.md) | Native OFT policy. |
| [`openvla-quic`](../../plugins/openvla-quic/README.md) | QuIC policy, training and checkpoint finalizer. |

Build recipes and their dependency locks are selected by the plugin registry.
Shared support code is installed through explicitly declared sources; the core
installer does not hardcode model package paths.

```bash
bash deploy/scripts/build-images.sh --dry-run
bash deploy/scripts/build-images.sh
```

Builds default to temporary source contexts acquired from pinned Git commits or
snapshotted local projects. They do not depend on mutable `external/` directories.
The optional `--workspace-sources` mode is for managed development checkouts. See the
[deployment guide](../README.md) for host preparation, selected roles, explicit
configuration bundling and runtime commands.

## Runtime boundaries

| Role | Runtime responsibility | Writable persistent storage |
| --- | --- | --- |
| Benchmark | LIBERO/Robosuite/MuJoCo, runner, RPC client | Canonical run evidence. |
| Policy | Model loading and inference | None. |
| Dataset | Acquire and prepare demonstrations | Managed dataset storage. |
| Training | Optimization and staging | The scoped training run directory. |
| Finalizer | Verify and publish a staged checkpoint | Checkpoint storage and scoped training evidence. |
| Reporting | Verify evidence, generate HTML and exports | Derived exports. |

Images do not contain model weights or datasets. Evaluation services communicate
over a private AF_UNIX socket, run with networking disabled, read-only root
filesystems and UID/GID `10001:10001`. Dataset acquisition is the separate online
phase. Checkpoints, datasets and optional adapters are mounted only into the
roles that need them, read-only where applicable. Reporting cannot modify
canonical evidence.

## Build and runtime identity

Production recipes pin base images by digest and install hash-locked artifacts
without dependency resolution or unlocked PEP 517 build isolation. They record
source, plugin, dependency and package-inventory metadata. OVLAB checks image role
and deployment/plugin compatibility before starting workloads. Rebuild when
runtime code, plugin implementations, upstream pins, locks or Docker recipes
change.

Configuration is a separate explicit closure. `config/` contains shipped OVLAB
defaults, plugin projects contain their own defaults, and `configs/` plus
`experiments/` contain ignored user content. Runtime composition snapshots only
selected references into a read-only bundle. Ordinary user YAML edits therefore
need no image rebuild. `--config FILE` can instead bake a selected complete
evaluation/deployment closure into a compatible image; its digest invalidates
the secret-consuming build layer.

## Verification

```bash
ovlab run examples/experiments/openvla-eval.yaml --plan
ovlab run examples/experiments/openvla-oft-eval.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
```

The last workflow verifies training, checkpoint handoff, evaluation and reporting
through the QuIC plugin on one GPU. It is deliberately too short to measure task
performance. Planning checks configuration without launching workloads.

For QuIC-specific mounts and role selection, see [QUIC.md](QUIC.md). Dependency
maintenance is described in [`../locks/README.md`](../locks/README.md). The
`Dockerfile.smoke` provider is test-only and never included in production images.
