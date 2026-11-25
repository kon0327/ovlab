# LIBERO plugin project

This project owns OVLAB's LIBERO benchmark and pinned RLDS dataset integration.
The Python distribution is `ovlab-libero`; its import package is `ovlab_libero`.
The generic benchmark SDK contains no LIBERO implementation.

The migration's current qualification status is recorded in
[EXTENSION_ARCHITECTURE.md](../../docs/EXTENSION_ARCHITECTURE.md). The layout and
contracts below describe the implementation; they do not establish that an
image build, GPU renderer or complete evaluation has passed qualification.

## Providers and configuration

`extension.json` uses `ovlab.extension/v1`, category `plugin`, and declares:

| Qualified provider | Contract | Purpose |
| --- | --- | --- |
| `plugin:benchmark:libero/libero` | `BenchmarkPlugin` API `"1"` | Benchmark configuration, tasks, capabilities, runtime adapter and deployment requirements. |
| `plugin:dataset:libero/rlds` | `DatasetProvider` API `"1"` | Resolve, acquire, verify and prepare the pinned training dataset. |

The repository's [common registry](../registry.json) binds the logical project
`libero` to this directory. A remote registration instead binds the project to
an immutable Git commit. This integration-project revision is independent of
the upstream simulator revision and of the RLDS dataset revision.

Use these project-owned references when composing an experiment:

```yaml
components:
  benchmark: plugin://libero/benchmark/default.yaml
```

The default benchmark configuration selects
`plugin:benchmark:libero/libero`. It leaves task subsets and experiment budgets
to the experiment. The native action contract is
`plugin://libero/interfaces/osc-pose-v1.yaml`. Renderer execution profiles are
`plugin://libero/execution/egl.yaml` and
`plugin://libero/execution/glfw.yaml`; they are independent of model settings.
User overrides remain explicit files under `configs/` or `experiments/` and
inherit these references.

## Project ownership

| Path | Contents |
| --- | --- |
| `extension.json` | Provider declarations, upstream source pin, configuration templates and image recipes. |
| `src/ovlab_libero/plugin.py` | Lightweight control hooks and lazy entrypoints into typed/runtime code. |
| `src/ovlab_libero/task_catalog.json` | Portable authoritative tasks tied to the pinned upstream revision. |
| `src/ovlab_libero/tasks.py` | Runtime native-task and initial-state bindings. |
| `src/ovlab_libero/adapter.py`, `environment.py` | Benchmark adapter and native environment lifecycle. |
| `src/ovlab_libero/observations.py`, `actions.py`, `signals.py` | Benchmark-owned canonical mappings and action acceptance. |
| `src/ovlab_libero/settings.py`, `renderer.py`, `validation.py` | Settings, execution choices and bounded configuration validation. |
| `src/ovlab_libero/dataset.py` | Pinned RLDS source provider. |
| `configs/` | Benchmark, action interface and execution defaults. |
| `runtime/config.yaml` | Native LIBERO path configuration used by its runtime image. |
| `docker/`, `locks/` | Benchmark/dataset image recipes and dependency locks. |

Large datasets, model weights and generated runs remain outside the project.
Evaluation BDDL, native assets and initial states come from the explicitly
pinned upstream LIBERO source. Runtime consumes prepared resources read-only;
it does not fetch replacement assets or silently normalize them.

## Native and scientific identities

The upstream LIBERO pin is
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. The task catalog preserves native
suite/task order and identifies its upstream revision. The plugin selects
portable task descriptors before native initialization, then verifies their
native bindings and actual capabilities in the worker.

The dataset provider separately selects `openvla/modified_libero_rlds` at
revision `a7c9ae18499b6eea8a32f78a9302327b752b1b5f`. That source revision, the
dataset resolution, verified file inventory and preparation identity describe
training data. They do not replace the benchmark source pin or identify a model
checkpoint.

New canonical runs record benchmark identity, selected task/suite identity and
execution metadata, including the resolved renderer. Consumers use those
fields instead of deriving meaning from task-ID prefixes. Historical evidence
keeps its existing bytes and schema; read compatibility does not rewrite it.

For observation profiles, exact action conventions, deterministic initial-state
selection, settling steps, signals and renderer behavior, see the
[adapter contract](src/ovlab_libero/README.md). Model-specific decoding,
normalization and action conversion remain owned by the selected policy project.

## Deployment and isolation

The benchmark provider declares its own image, build role, renderer environment,
GPU requirements and optional mounts. A separate policy provider declares the
model service. OVLAB combines the selected pair through the shared Compose
template and verifies each image against its own sources and recipe; a model
project does not choose the benchmark's runtime requirements.

Declaration discovery does not import MuJoCo, Robosuite, Torch or LIBERO. Typed
capability/catalog work can load the canonical SDK without starting a native
environment. The worker applies renderer settings before importing native
libraries and validates actual capabilities after initialization.

The benchmark and dataset recipes are `docker/Dockerfile` and
`docker/Dockerfile.dataset`; their qualified build roles are `libero/benchmark`
and `libero/dataset`. The project source is staged at
`external/extensions/libero/self`, with the upstream at
`external/extensions/libero/libero`. The public project commands inspect their declared
inputs without requiring a benchmark execution:

```bash
./ovlab plugin inspect libero
./ovlab plugin check libero
./ovlab plugin build libero --dry-run
```

`check` includes supported lightweight control checks; it is not evidence of a
working renderer, native runtime, dataset acquisition or full evaluation. An
image/build plan is not an image qualification. The full acceptance path also
requires clean installed/frozen execution, an actual native LIBERO smoke,
canonical integrity verification, metric recomputation and report/export
verification. See [dataset operations](../../DATASETS.md) for explicit resource
acquisition and the architecture evidence log for completed gates.
