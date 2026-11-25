# ovlab-benchctl

OVLAB's composition and orchestration layer owns configuration resolution,
experiment scheduling, checkpoint acquisition, container lifecycle and evidence.
Model implementations and their runtime dependencies belong to registered plugin
projects. Plugins connect through the policy SDK; core configuration does not
select a concrete model implicitly.

## Configuration ownership

| Reference | Owner |
| --- | --- |
| `ovlab://metrics/default.yaml` | Tracked OVLAB defaults under `config/`. |
| `plugin://openvla/policy/default.yaml` | Defaults in a registered plugin project. |
| `policy/my-model.yaml` | An explicit user extension under ignored `configs/`. |
| `experiments/my-study.yaml` | An ignored user experiment selected by path. |

Plugin defaults use separate `policy`, `deploy`, `registry`, `train` and
`evaluate` categories. Train/evaluate defaults may be partial: experiments add
dataset, benchmark, metric, protocol and artifact choices before validation.
The maintained verification scenarios live in `examples/experiments/`.

The strict YAML subset rejects duplicate/unknown keys, anchors, aliases, tags,
multi-document files and reference traversal. Relative `extends` stays within
its configuration origin; explicit `plugin://` and `ovlab://` references cross
origins. Mappings merge recursively; sequences and scalars replace their parent.

`ConfigResolver.resolve()` composes complete documents, resolves checkpoint,
device and artifact resources through an explicitly selected machine profile,
and checks action/observation compatibility. Scientific hashes exclude resolved
machine paths/devices and renderer choices; execution hashes include them.
Renderer profiles are `plugin://libero/execution/egl.yaml` and
`plugin://libero/execution/glfw.yaml`. `MUJOCO_GL` and `MUJOCO_EGL_DEVICE_ID` remain
explicit diagnostic overrides and enter execution provenance.

Docker deployments freeze the selected transitive configuration closure and
required plugin control code into a deterministic read-only bundle. Both workers
receive the same bundle. Checkpoints are verified separately and mounted
read-only; large weights stay outside plugin Git projects and runtime images.

## CLI and developer workflow

From the repository root:

```bash
./ovlab plugin list
./ovlab plugin inspect openvla
./ovlab plugin check openvla
./ovlab run examples/experiments/openvla-eval.yaml --plan
./ovlab run examples/experiments/openvla-oft-eval.yaml --plan
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml --verify
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml
./ovlab run EXPERIMENT_ID_OR_HASH --verify
./ovlab report EXPERIMENT_ID_OR_HASH
./ovlab report recompute EVALUATION_RUN_ID
./ovlab data list runs --detail
./ovlab deploy run plugin://openvla/deploy/default.yaml --dry-run
```

`ovlab run` accepts a training profile, evaluation document or hierarchical
workflow. `--plan` and configuration `--verify` resolve and validate without
starting workloads. Verification of a stored experiment checks its canonical
phase evidence and checkpoint bindings. Persistent model services use `deploy`;
`deploy serve` runs a foreground service and `deploy connect` checks its
compatibility with an explicit experiment. Plugin commands select project IDs;
variants and supported options appear in `plugin inspect`. `report recompute`
audits recorded episode metrics without modifying canonical data. See
[CLI_README.md](CLI_README.md) for selectors, transport and configuration modes.

`OVLAB_LOCAL_PROFILE` or the explicit profile option selects machine settings;
the fallback is ignored `configs/local/profile.yaml`. `OVLAB_EXECUTION_PROFILE`
can select a renderer profile. `ResolvedExperimentConfig.write()` writes a
deterministic resolved configuration and refuses to overwrite existing evidence.

Human output is the default. `--json` writes one versioned JSON document to
stdout; diagnostics use stderr. Exit codes are:

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 2 | Command-line usage error |
| 3 | Configuration or validation error |
| 4 | Policy/provider unavailable or incomplete |
| 5 | Service, protocol or capability error |
| 6 | Benchmark/runtime failure |
| 7 | Run/trace integrity failure |
| 8 | Offline metric recomputation failure |
| 130 | Interrupted by SIGINT or SIGTERM |

Install the host environment following [ENVIRONMENT.md](../../../ENVIRONMENT.md).
The [plugin contract](../../../docs/POLICY_PLUGINS.md) documents extension hooks;
[the workflow guide](../../../docs/EXPERIMENT_WORKFLOW.md) describes composition,
checkpoint handoff, scheduling and immutable evidence.
