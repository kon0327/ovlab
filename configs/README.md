# OVLAB configuration

OVLAB composes experiments from explicit, versioned component documents. There
is no implicit search path, Hydra defaults list, environment-variable expansion,
or runtime mutation. An experiment names every component, the resource registry,
and—when it is deployable—its portable deployment topology and renderer. A
machine-local resource profile is selected separately by the caller.

Inheritance is opt-in through a relative `extends` path. Mappings deep-merge,
while scalar and sequence values replace the parent. Every composed document is
validated against an exact schema, so misspelled or unknown keys fail early.

The policy's `settings.action.action_interface` selects the action contract.
The shared LIBERO interface records the verified `closed_positive` gripper
convention. The benchmark's `settings.action.interface_ref` must resolve to the
same interface, and the resolver compares it with the concrete LIBERO adapter
contract. Canonical camera names are checked in the same way.

## Scenario directories

- `experiments/training/`: training scenarios selected by `ovlab train run TARGET`.
- `experiments/evaluation/`: evaluation scenarios selected by `ovlab eval run TARGET`.
- `deploy/`: policy-only deployment configs selected by `ovlab deploy run TARGET`.

Each accepts a YAML name or path and supports `--dry-run`. Training/evaluation
`inspect` and `verify` take run references; deployment uses service IDs or config
names. See [`../DEPLOYMENT.md`](../DEPLOYMENT.md) for deployment schema, image
building, ports, quantization, readiness, and service lifecycle.

## LIBERO renderer execution profiles

Renderer selection is declared in the execution-only `deployment` block of a
deployable experiment. `renderer: egl` selects
`profiles/libero-bench-egl.yaml` for headless benchmarking; `renderer: glfw`
selects `profiles/libero-playground-glfw.yaml` for an interactive playground.
The selected profile and resolved renderer are included in the execution
configuration hash; changing them leaves the scientific hash, tasks, seeds,
observation/action contracts, and metrics unchanged. The CLI's `--renderer`
option and the lower-level `OVLAB_EXECUTION_PROFILE` diagnostic override take
precedence over the experiment default.

The EGL profile resolves its device from the local profile's optional
`execution.libero.renderer.device_id`, falling back to the index in
`devices.primary_gpu`. GLFW never emits an EGL device setting. Diagnostic
process environment values have precedence: `MUJOCO_GL` selects `egl` or
`glfw`, and `MUJOCO_EGL_DEVICE_ID` overrides the EGL device. Unsupported values
fail during configuration resolution. No MuJoCo, Robosuite, or LIBERO import is
performed by the resolver.

Machine paths and devices belong in gitignored `local/*.yaml` profiles. The
resolver produces a scientific hash without that profile and an execution hash
including all resolved paths and devices. Its immutable output is one
`resolved_config.yaml`.

## Reporting and export configuration

An experiment may declare a `reporting` block with `enabled`, a built-in or
local `profile`, task/run finalization hooks, and `failure_policy`. Reporting
defaults to the built-in `libero-task-default` profile when the block is omitted.
Its content affects only the derived report identity, not the scientific or
benchmark execution configuration hashes.

New exports use built-in, versioned `isolated-default-v1` and
`grouped-default-v1` templates selected directly by the CLI. Their normalized
selection and every source run's canonical configuration identity are recorded
in the resulting `metadata.json`. Files below `exports/` retain only the legacy
`ovlab.export-spec/v1` grouped bridge; new grouped comparisons should use
`ovlab export grouped`. Python expressions, YAML tags, arbitrary imports, and
remote assets are not supported.

## Checkpoint resources

Portable policy configurations select a logical `settings.checkpoint_id`.
`resources/registry.yaml` binds that ID to its repository, immutable `revision`,
aggregate SHA-256, and file manifest. Absolute paths never belong in an
experiment or the portable registry.

A gitignored local profile may provide
`resources.checkpoints.<checkpoint-id>.local_path` for an unpublished artifact
such as a QuIC checkpoint. The path is execution-only. Repository, revision,
file sizes, and hashes remain registry-controlled and therefore remain in the
scientific configuration hash.

Docker deployment resolves global Hugging Face cache, local override, and
OVLAB-managed cache in that order. The policy always sees the selected artifact
at `/checkpoints/resolved/<checkpoint-id>` and cannot observe its host location.

Model-family policy owners validate their own deployment and artifact contracts.
QuIC adapter-only training and placement studies live under `experiments/training/` and
`ablations/`; benchmark experiments continue to select policies and benchmarks
independently.

## Training and ablation profiles

`experiments/training/` uses `ovlab.training-profile/v1` with the same explicit relative
`extends` merge rules. `ablations/` uses `ovlab.training-ablation/v1` and references
a common training base plus ordinary variant profiles. Declared method factor
paths may differ; checkpoint/data identities, episode split, normalization, seed,
optimizer budget and resource contract must remain shared.

```bash
./ovlab train ablate validate --suite configs/ablations/openvla-quic-placement-smoke.yaml
./ovlab train ablate materialize --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-plan
```

Materialization resolves and verifies local data/model resources and creates a
new immutable plan plus profiles. `train ablate run` executes those profiles
through the normal isolated training/checkpoint/report lifecycle. See
[`../docs/ABLATIONS.md`](../docs/ABLATIONS.md) for the episode partition contract
and source-project migration audit.
