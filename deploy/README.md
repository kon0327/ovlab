# OVLAB deployment

OVLAB resolves an experiment, starts isolated policy and benchmark containers,
finalizes canonical evidence, and publishes reports. Model plugins own their
runtime images, upstream sources, training hooks and checkpoint catalogs. OVLAB
owns the scheduler, container lifecycle, RPC transport, configuration composition
and reporting.

## Configuration ownership

| Location | Contents |
| --- | --- |
| `config/` | Versioned OVLAB defaults, referenced as `ovlab://...`. |
| `plugins/<model>/configs/` | Model defaults, referenced as `plugin://<model>/...`. |
| `configs/` | Ignored local extensions, machine profiles and plugin registrations. |
| `experiments/` | Ignored personal experiment recipes. |
| `examples/experiments/` | Small versioned verification scenarios. |

An experiment explicitly selects extensions. The build and runtime resolvers
follow its `extends`, component, checkpoint-registry and execution-profile
references; unrelated local YAML is not imported. Mappings merge recursively,
while lists and scalars replace their parent values. Plugin `train` and
`evaluate` defaults are composable fragments: the experiment adds dataset,
benchmark, metrics and reporting choices before validation.

The independently registered model projects are `openvla` (Vanilla and LoRA),
`openvla-oft`, and `openvla-quic`. Shared implementation support is an explicit
plugin dependency, not a model dispatch table in the OVLAB core. See the
[plugin contract](../docs/POLICY_PLUGINS.md) for local and pinned Git registration.

## Build images

From the repository root:

```bash
bash deploy/scripts/build-images.sh --dry-run
bash deploy/scripts/build-images.sh
```

The default build prepares a temporary context from registered sources. Git
sources use immutable commits; local projects are snapshotted. Existing
`external/` development checkouts are neither required nor modified. Use plugin
source synchronization only when you need those development copies. An explicit
`--workspace-sources` build instead uses the managed development workspace and rejects
unregistered or dirty Git checkouts.

Build only selected roles while iterating:

```bash
bash deploy/scripts/build-images.sh benchmark reporting dataset
bash deploy/scripts/build-images.sh policy-openvla
bash deploy/scripts/build-images.sh policy-openvla-oft
bash deploy/scripts/build-images.sh policy-openvla-quic training-openvla-quic finalizer-openvla-quic
```

Runtime recipes and locks live in the project that owns each model. Infrastructure
recipes remain in [`docker/`](docker/README.md). All builds use pinned base image
digests, locked Python artifacts and source provenance. Model weights and datasets
are acquired separately and mounted at runtime.

An ordinary build does not include ignored user configurations. To bake an
explicit, complete evaluation or deployment configuration and its closure into an
image, use the build script's `--config FILE` option. A workflow is resolved into
its phase configurations by `ovlab run`; do not pass a partial plugin evaluation
fragment as a standalone build configuration. The configuration bundle digest is
part of the build cache key.

## Prepare data and host settings

Docker Engine, Compose v2, the NVIDIA Container Toolkit and a compatible GPU driver
are required for the GPU examples. EGL is the default headless LIBERO renderer.
Keep checkpoints, datasets and evidence outside the checkout, for example in a
sibling `ovlab-data/` directory. See [data preparation](../DATASETS.md) and
[checkpoint handling](../DATA_MANAGEMENT.md).

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Set machine-specific cache/data roots, GPU and EGL device selections in that
ignored file. `OVLAB_RUNS_ROOT` selects canonical evidence storage;
`OVLAB_EXPORTS_ROOT` selects derived reports and exports. The CLI normally creates
required artifact directories and supplies the invoking user's supplementary GID
for non-root container access. Direct Compose diagnostics must provide equivalent
permissions for UID/GID `10001:10001`.

The host resolves a logical checkpoint ID from the selected catalog, verifies its
revision and file inventory, and mounts only that resolved snapshot read-only.
Policy containers load it offline. An experiment may explicitly select a local
checkpoint or use `checkpoint_from` to consume a finalized training output.

## Plan and execute

```bash
ovlab run examples/experiments/openvla-eval.yaml --plan
ovlab run examples/experiments/openvla-oft-eval.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
ovlab run examples/experiments/plugin-pipeline-smoke.yaml
```

The pipeline smoke performs three QuIC optimizer steps on a bounded task subset,
finalizes the adapter, evaluates that exact checkpoint for one two-step rollout,
and generates training/evaluation reports. It is configured for a single 24 GiB
GPU and tests integration rather than model quality. The model-free mock example
can be inspected with `--plan`; these deployment instructions cover the plugin
GPU execution path.

Create personal variants under ignored `experiments/` and select ignored component
extensions under `configs/`. For an offline or GLFW variant, set
`config.evaluate.patch.deployment.offline: true` or
`config.evaluate.patch.deployment.renderer: glfw` in that workflow. GLFW needs an
existing display server; the launcher passes the host display and read-only X11
socket to the benchmark.

## Isolation and artifacts

During evaluation, policy and benchmark services communicate over an AF_UNIX
socket in a project-owned volume. They have no published ports, no network,
read-only root filesystems, dropped Linux capabilities and no Docker socket. The
policy receives verified checkpoint/adapter mounts; the benchmark receives
read-only datasets and write access to canonical run storage. Reporting receives
canonical storage read-only and a separate writable exports root.

Training and finalization use separate plugin-owned roles. The trainer reads its
base checkpoint and prepared dataset and writes staged training evidence. A CPU
finalizer verifies and publishes the checkpoint; the trainer does not write
directly into the checkpoint catalog. Reporting also runs separately.

Workflow instances store phase evidence under `runs/<experiment-instance>/` and
derived output under `exports/<experiment-instance>/`. The canonical evidence
records resolved configurations, plugin/source identities and checkpoint binding.
Runtime teardown and reporting failure preserve completed canonical evidence.

```bash
ovlab run EXPERIMENT_INSTANCE_ID --verify
ovlab report EXPERIMENT_INSTANCE_ID
```

`ovlab run` performs project-scoped cleanup, including its RPC volume, on success,
failure or interruption. It does not delete canonical data or unrelated Docker
resources. Use scoped Docker inspection for troubleshooting; global prune is not
part of the workflow.

## Diagnostics

Checkpoint failures report the selected logical ID, catalog and verification
stage. With offline mode enabled, missing cached artifacts are hard failures.
Policy startup errors usually identify missing weights, CUDA/FlashAttention
compatibility or an unsupported plugin configuration. EGL errors require checking
graphics-device access; GLFW additionally requires the display server. Filesystem
permission errors require correcting host ownership or ACLs rather than making
artifact directories world-writable.

The test-only [transport smoke](scripts/transport-smoke.sh) exercises container
IPC and startup without loading a model. Its provider exists only in the smoke
image and is excluded from production roles. It provides transport evidence,
not benchmark or model-quality evidence.
