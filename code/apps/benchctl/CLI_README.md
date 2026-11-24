# OVLAB CLI

`ovlab` orchestrates registered plugins, explicit configurations, isolated
execution and immutable results. It does not implement model inference,
benchmark loops or metric formulas. Run `./ovlab` from a checkout, or `ovlab`
after installing `ovlab-benchctl`. Use `--help` on any command for its options.

## Public commands

| Command | Responsibility |
| --- | --- |
| `editor` | Open the local visual workflow editor. |
| `doctor` | Diagnose host, Docker and optional experiment readiness without changing files. |
| `plugin` | Register, inspect, check, sync and build plugin projects. |
| `config` | Validate or resolve composed configurations. |
| `deploy` | Serve, deploy, connect to, inspect, verify and stop model services. |
| `run` | Plan, execute or verify training, evaluation and complete workflows. |
| `report` | Generate reports/exports, verify reports or recompute recorded metrics. |
| `dataset` | Resolve, acquire, prepare and verify immutable datasets. |
| `checkpoint` | Inspect registered/local checkpoints and verify their contents. |
| `data` | List, archive or delete stored runs, reports and exports. |

`policy`, top-level `connect`, `service` and `metrics` have been removed.
Phase-specific execution commands and `_worker` are internal interfaces;
normal execution uses `run`, and service management uses `deploy`.

## Output and errors

Structured commands accept mutually exclusive output options:

- Default: compact tables and summaries.
- `--detail`: complete readable JSON.
- `--json`: one `ovlab-cli-output/1.0.0` envelope for scripts.

Progress and diagnostics use stderr. `config resolve` prints the resolved
configuration directly, with `--format yaml|json`.

Exit codes are 0 for success, 2 for invalid arguments, 3 for invalid
configuration, 4 for unavailable policy/resources, 5 for service failures,
6 for runtime failures, 7 for integrity failures, 8 for metric errors, and
130 for interruption. A successful compatibility probe can still report
`compatible: false`; inspect that field when using it in automation.

## Register and check a plugin project

```bash
./ovlab plugin register /path/to/my-plugin
./ovlab plugin register https://github.com/owner/plugin.git --commit FULL_GIT_SHA
./ovlab plugin list
./ovlab plugin inspect openvla
./ovlab plugin check openvla
```

Project IDs select complete registrations. `plugin list` displays each project's
kind, version, source, pinned commit (for Git) and variants. `inspect` displays
all variants, declared capabilities, configuration templates and source
identities. Variant names such as `vanilla` and `lora`, and internal policy
types, are not project IDs.

`plugin check` validates the manifest, reads declared configuration templates,
and loads the SDK control entrypoints to check API compatibility. It does not
instantiate a runtime adapter, load weights, run inference, or prove runtime
readiness. Incomplete configuration templates remain templates; validate the
completed experiment separately. Control hooks must stay independent of model
runtime libraries. Use `--offline` to require a previously cached Git project.

Registrations normally go into ignored `configs/plugins/registry.json`.
`--registry PATH` selects another registry for a plugin command;
`OVLAB_PLUGIN_REGISTRY` selects it for subsequent commands too. See
[plugin development](../../../docs/POLICY_PLUGINS.md).

## Build images

```bash
./ovlab plugin build openvla --dry-run
./ovlab plugin build openvla
./ovlab plugin build openvla --config configs/evaluate/my-model.yaml
```

A project build selects its declared policy, training and finalizer recipes.
Omit the project ID to build every registered project's recipes. To build
infrastructure or individual roles, use
`bash deploy/scripts/build-images.sh ROLE ...`; no roles builds everything.

Builds fetch pinned Git sources or copy registered local sources into a temporary
context. They do not depend on existing `external/` checkouts. Optional
`plugin sync PROJECT` prepares a development workspace; `--workspace-sources`
explicitly builds from that workspace. `--offline` rejects missing sources.

## Compose and validate configuration

Plugin defaults use `plugin://PROJECT/path.yaml`; core defaults use
`ovlab://path.yaml`. User extensions are explicit files in ignored `configs/`
and experiments in ignored `experiments/`. Installing a project makes its
configurations available without copying them into core directories.

```bash
./ovlab config resolve plugin://openvla/policy/default.yaml
./ovlab config validate configs/evaluate/my-model.yaml
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
```

Partial plugin defaults must be completed by an explicit experiment or override.
`config validate --mode runtime` additionally requires the selected local
resources and profile. Local profiles stay outside version control; select one
with `OVLAB_LOCAL_PROFILE=configs/local/profile.yaml`.

## Execute and verify

```bash
cp deploy/compose/.env.example deploy/compose/.env
./ovlab doctor
./ovlab doctor --experiment configs/evaluate/my-model.yaml
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml
./ovlab run EXPERIMENT_OR_RUN_ID --verify
```

`run` accepts explicit YAML, workflow JSON, a plugin configuration reference or
a unique experiment name. Schema and kind select the phase. `--plan` resolves
without starting workloads; `--verify` validates a configuration or verifies
stored experiment/run artifacts. Model, renderer, quantization, scheduling and
checkpoint choices belong in configuration.

A workflow can bind evaluation to its finalized training checkpoint. Evaluation
can also select a plugin catalog checkpoint or a registered user checkpoint.
Large weights remain in external data storage and are mounted read-only.

Evaluation uses a validated, hashed configuration bundle shared by policy and
benchmark containers. User YAML edits normally do not require rebuilding images;
packaged code, source revisions and locked dependencies do. Containers and private
RPC volumes are removed after an evaluation; canonical run data remains.

## Deploy and connect

Persistent Docker deployment:

```bash
./ovlab deploy run configs/deploy/my-service.yaml --dry-run
./ovlab deploy run configs/deploy/my-service.yaml
./ovlab deploy inspect SERVICE_ID
./ovlab deploy verify SERVICE_ID
./ovlab deploy connect SERVICE_ID --experiment configs/evaluate/my-model.yaml
./ovlab deploy stop SERVICE_ID
```

`verify` checks the recorded image/configuration identity and service readiness.
`connect` checks the running policy's handshake and capability compatibility
against an explicit evaluation configuration, without prediction or rollout.
A managed target must have a published TCP port. Direct endpoints also work:

```bash
./ovlab deploy connect --experiment EVAL.yaml --socket /tmp/policy.sock
./ovlab deploy connect --experiment EVAL.yaml --host 127.0.0.1 --port 8000
```

Without a managed target or explicit endpoint, the experiment's default Unix
socket is used. A managed target and direct endpoint flags are mutually
exclusive. The benchmark describes its configured capabilities through
`describe_capabilities(action_spec)` without starting a simulator.

For native development, or inside an image with its runtime dependencies:

```bash
./ovlab deploy serve DEPLOYMENT_OR_EVAL.yaml --socket /tmp/policy.sock
./ovlab deploy serve DEPLOYMENT_OR_EVAL.yaml --port 8000 --persistent
```

`serve` stays in the foreground. `--persistent` keeps the model ready between
client sessions. Container healthchecks use the internal `_worker health`
interface; users normally use `deploy verify`.

## Reports and metric audit

```bash
./ovlab report RUN_OR_EXPERIMENT_ID
./ovlab report profiles
./ovlab report verify RUN_ID
./ovlab report recompute RUN_ID --json
```

`report recompute` verifies the saved evaluation run, recomputes episode metrics
using its recorded metric configuration, and compares them with the original
results. It needs neither a model nor a simulator and never modifies canonical
traces, recorded metrics or existing reports. It is an audit operation, not a
way to apply a new metric profile to an old run. Inspect `all_results_agree`
and per-episode `comparisons` in the output.

The checkout launcher uses the reporting image by default. Recompute runs with
network disabled and read-only run data; it needs no existing export directory.
`OVLAB_REPORTING_RUNTIME=host` explicitly selects an appropriately prepared host
Python environment for diagnostics. Rebuild images after changing packaged CLI
code, including when migrating from the removed command names. Older reporting
images are rejected with a rebuild command before starting a metric audit.

See [deployment](../../../DEPLOYMENT.md),
[environment setup](../../../ENVIRONMENT.md) and
[plugin architecture](../../../docs/POLICY_PLUGINS.md) for details.
