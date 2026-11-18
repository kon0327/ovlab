# OVLAB CLI

`ovlab run EXPERIMENT` executes a training profile, evaluation scenario, or
hierarchical workflow through one experiment owner. Add `--plan` to inspect the
resolved configuration tree or `--verify` to validate a source configuration.
`ovlab run INSTANCE_ID_OR_HASH --verify` checks a stored experiment and every
canonical phase run. Individual run IDs/hashes also work with `--verify`.
Runtime settings live in YAML. Phase-specific commands are private compatibility
interfaces used by workers; they are not public run subcommands. Persistent model
services use `ovlab deploy`; see [model deployment](../../../DEPLOYMENT.md).

`ovlab` is the unified command-line interface for OpenVLABenchmark. It exposes
the existing configuration, policy-service, runner, artifact, and metric APIs
through one workflow:

```text
Dataset -> Train -> Checkpoint -> Eval -> Verify -> Report
                             -> Deploy (persistent model service)
```

The CLI is an orchestration layer. It does not implement policy inference,
benchmark loops, metric formulas, action conversion, or scientific hashing.

## Invocation

From the repository root, use the installation-independent launcher:

```bash
./ovlab --help
```

When `ovlab-benchctl` is installed, the same interface is available as:

```bash
ovlab --help
```

The source-tree launcher does not install or update packages. `deploy` commands
need only system Python 3, Docker Engine and Docker Compose; they do not require
an activated Conda environment.

Open the static workflow editor in the host's default browser with:

```bash
ovlab editor
```

This works from any directory. On WSL, it converts the editor path with
`wslpath` and opens it using the default Windows file association. On Linux it
uses `xdg-open`.
The command does not start a web server or Docker container. See the
[Workflow Studio guide](../workflow-builder/README.md) for editing and catalog
refresh instructions.

## Output modes

Structured commands use one consistent output contract:

- no output flag: labeled, width-aware tables for lists and `key  value` summaries for other results;
- `--detail`: the complete result document as readable indented JSON;
- `--json`: one stable `ovlab-cli-output/1.0.0` JSON envelope for scripts.

`--detail` and `--json` are mutually exclusive. Operational progress and
diagnostics remain on stderr, so `--json` stdout stays machine-readable.
Tables wrap long values to fit the terminal; on very narrow terminals they use
vertical records. Use `--detail` or `--json` when copying complete paths or
identifiers. The `config resolve` command is intentionally different: its result
is the complete resolved configuration and `--format yaml|json` selects its serialization.

## Recommended Docker workflow

Check the host before deploying:

```bash
./ovlab doctor
./ovlab doctor --experiment experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml
./ovlab doctor --experiment experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml --json
```

`doctor` reads the checkout and Compose environment, probes Docker Compose and
the Docker daemon, and reports a fix for each failed check. With `--experiment`
it also validates the configuration bundle, checks host artifact paths, and
inspects the selected images for the current deployment contract label. It
does not create directories, start containers, download checkpoints, or load a
model. Warnings return exit code 0; failed checks return exit code 6. JSON
output retains every check in `result.checks`, including on failure. Use
`--env-file PATH` to inspect a different Compose environment file inside the
checkout.

Copy the deployment environment template once and set the host paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Then run a complete isolated policy-service plus LIBERO benchmark topology with
one command:

```bash
./ovlab run \
  experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml
```

`EXPERIMENT` accepts an explicit YAML path, a workflow JSON path, or a unique
name below `experiments/`. Its schema and kind select the phase automatically.

The CLI validates the Compose model, passes the same experiment to both
containers, waits for policy readiness, runs the benchmark, propagates failure,
and removes its containers and private RPC volume. Canonical artifacts remain in
the host directory configured by `OVLAB_RUNS_ROOT`.

The selected experiment is not baked into an image. For every deployment the
CLI validates and hashes the exact transitive YAML closure, materializes it as a
temporary read-only bundle, and mounts the same bundle into benchmark and policy
containers at `/opt/ovlab/configs`. The bundle is removed after teardown while
the portable and resolved configuration remains in the canonical run artifacts.
Adding or changing experiment YAML therefore does not require an image rebuild;
changing packaged Python code, locked dependencies or external source does.

Each run directory uses the readable host-local naming contract
`<experiment-id>_YYYY-MM-DD_HH-MM-SS_<8-char-hash>`. The manifest retains the
authoritative UTC nanosecond timestamp; the path timestamp is intended for
operators and the short hash prevents collisions between runs started within
the same second.

LIBERO protocols record every canonical primary-camera observation by default.
Finalization creates `video.mp4` and checksum-bearing `video.json` inside every
episode directory; no timestep subsampling is applied. The original RGB arrays
remain part of the immutable trace, so video provenance and frame counts can be
verified offline.

LIBERO datasets follow the same external-data convention. If
`OVLAB_DATASETS_PATH` is unset, deployment derives
`<OVLAB_RUNS_ROOT>/../datasets/libero`, creates the directory, and mounts it
read-only. An explicitly configured dataset path must already exist.

Every deployable experiment explicitly declares its Compose topology and renderer:

```yaml
deployment:
  profile: openvla  # openvla | oft
  renderer: egl     # egl | glfw
```

OpenVLA Vanilla and merged OpenVLA-LoRA both use `profile: openvla` because
they share the same policy image and Compose service. LoRA remains a distinct
scientific method through its policy component; creating a duplicate Compose
profile would not add isolation. OpenVLA-OFT uses `profile: oft`.

Set `deployment.renderer` in the evaluation YAML (`egl` or `glfw`). The Compose
profile must match the selected policy type. GLFW requires an interactive display.
Preview the resolved tree with:

```bash
./ovlab run CONFIG --plan
```

Inference quantization is `settings.runtime.quantization` in the policy component
(`none`, `8bit`, or `4bit`). A workflow can set it through an inherited
`config.evaluate.component_patches.policy` mapping. QuIC requires `none`.

Run the five-task cross-suite Vanilla NF4 experiment with:

```bash
./ovlab run \
  experiments/evaluation/libero10-openvla-vanilla-4bit-five-episodes.yaml
```

The selected policy configuration determines the inference representation and
its scientific identity; model/runtime choices are not CLI flags.

Before Docker starts, the CLI resolves the policy's portable `checkpoint_id`.
It checks `~/.cache/huggingface`, an optional local-profile `local_path`, and
`ovlab-data/checkpoints/huggingface`; if still absent, it downloads exactly the
registry-pinned revision into managed storage. File sizes and SHA-256 values are
verified before the snapshot is mounted read-only. Resolution, per-file download
progress, transferred byte counts, and verification stages are printed to
stderr; `--json` stdout therefore remains machine-readable. Set `deployment.offline: true` to prohibit checkpoint downloads. Training uses
explicitly prepared offline resources. Evaluation resolves the checkpoint
revision selected by its policy and resource registry.

Custom unpublished checkpoints remain outside the repository and are selected
through a gitignored local profile. Compose uses `deploy/compose/.env` by default;
`OVLAB_COMPOSE_ENV_FILE` selects another repository-local environment file:

```bash
OVLAB_LOCAL_PROFILE=configs/local/profile.yaml ./ovlab run CONFIG
```

## Machine-local native configuration

Low-level native development commands require a local profile containing
host-specific paths and device selection:

```bash
export OVLAB_LOCAL_PROFILE=configs/local/profile.yaml
```

If the variable is unset, the CLI looks for `configs/local/profile.yaml`.
Local profiles should remain gitignored. Portable experiment files must not
contain absolute paths or host-specific device assumptions.

For diagnostics, a LIBERO execution profile can be selected with:

```bash
export OVLAB_EXECUTION_PROFILE=configs/profiles/libero-bench-egl.yaml
```

Use `ovlab report RUN_ID` to export one canonical run, or `ovlab report
EXPERIMENT_ID` to export all runs in a completed experiment. Each run gets an
isolated export at its node. Each parent with at least two child branches holding
runs of the same category also gets a grouped export of those descendants.
Training and evaluation are handled separately. The output lists each node,
export, source runs and destination. Legacy report and export commands remain
available for compatibility with existing automation.

## Command overview

```text
ovlab
├── --help
├── -v, --version
├── doctor [--experiment EXPERIMENT] [--env-file PATH] [--detail | --json]
├── config
│   ├── validate CONFIG [--mode descriptor|runtime] [--detail | --json]
│   └── resolve CONFIG [--mode descriptor|runtime] [--format yaml|json]
├── policy
│   ├── list [--detail | --json]
│   └── describe CONFIG [--detail | --json]
├── service
│   ├── serve CONFIG [--socket PATH | --host HOST --port PORT] [--persistent] [--detail | --json]
│   └── health (--socket PATH | --host HOST --port PORT) [--detail | --json]
├── connect CONFIG [--detail | --json]
├── run EXPERIMENT [--plan | --verify] [--detail | --json]
├── deploy
│   ├── run TARGET [--profile LOCAL_PROFILE] [--quantization none|8bit|4bit]
│   │   [--port PORT] [--offline] [--dry-run] [--detail | --json]
│   ├── inspect TARGET [--detail | --json]
│   ├── verify TARGET [--detail | --json]
│   └── stop TARGET [--dry-run] [--detail | --json]
├── metrics recompute RUN_PATH|RUN_ID|RUN_HASH [--detail | --json]
├── report RUN_ID|EXPERIMENT_ID [--detail | --json]
├── report publish RUN_ID [--profile PROFILE] [--task TASK_ID] [--output PATH] [--detail | --json]
├── report profiles [--detail | --json]
├── report verify --run RUN_ID|RUN_HASH [--profile PROFILE] [--build BUILD_ID] [--detail | --json]
├── data
│   ├── list [--kind runs|reports|all] [--archived] [--detail | --json]
│   ├── archive (--run RUN_REFERENCE | --report REPORT_REFERENCE | --all) [--force] [--dry-run] [--yes] [--detail | --json]
│   └── delete (--run RUN_REFERENCE | --report REPORT_REFERENCE | --all) [--dry-run] [--yes] [--detail | --json]
├── dataset
│   ├── providers [--detail | --json]
│   ├── resolve --benchmark libero --suite SUITE [--detail | --json]
│   ├── fetch --source libero|url --name NAME [URL OPTIONS] [--detail | --json]
│   ├── import --name NAME --version VERSION --path PATH [--detail | --json]
│   ├── prepare --dataset DATASET_ID --format FORMAT [--detail | --json]
│   ├── list [--detail | --json]
│   ├── inspect --dataset DATASET_ID [--detail | --json]
│   └── verify --dataset DATASET_ID [--detail | --json]
└── checkpoint
    ├── list [--detail | --json]
    ├── export CHECKPOINT_ID --format bundle|tar --output PATH [--detail | --json]
    ├── inspect --checkpoint CHECKPOINT_ID [--detail | --json]
    └── verify --checkpoint CHECKPOINT_ID [--detail | --json]
```

Dataset and training commands use separate least-privilege images and never
write benchmark runs. Acquisition is explicit; validation, planning and all
inspection commands are offline and do not initialize a model. See
[`TRAINING.md`](../../../TRAINING.md) for the storage, identity, interruption
and base-plus-adapter deployment contracts.

## 1. Validate and resolve configuration

Descriptor validation checks configuration identity and structure without
requiring a runnable provider:

```bash
./ovlab config validate \
  experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml \
  --mode descriptor
```

Runtime validation additionally requires a complete runnable configuration:

```bash
./ovlab config validate \
  experiments/evaluation/libero10-lora-merged-rpc-smoke.yaml \
  --mode runtime
```

Print deterministic resolved configuration as YAML or JSON:

```bash
./ovlab config resolve CONFIG --mode runtime --format yaml
./ovlab config resolve CONFIG --mode runtime --format json
```

Resolution uses the existing OVLAB resolver. The CLI does not introduce
dot-list overrides or a second configuration hierarchy.

## 2. Inspect policies

List registered policy identities without loading models or discovering
runtime providers:

```bash
./ovlab policy list
./ovlab policy list --json
```

The catalogue describes implementation availability and source ownership, not
whether a model or its dependencies are installed or ready to run. The QuIC-PEFT
row lists OVLAB's supported training profiles, QP1–QP4 and `custom`; the legacy
QP0 descriptor does not represent an active transformation. QuIC-WC is an
unavailable proposed runtime. `config_type` for QuIC-WC names its descriptor
variant, while QuIC-PEFT uses the runnable policy configuration type.

Describe the policy selected by an experiment:

```bash
./ovlab policy describe CONFIG
```

The description reports the method family, configured artifact, capabilities,
readiness, and available scientific and execution identities. It does not load
a model or initialize CUDA.

## 3. Low-level native policy service

This command is an internal container entrypoint and a developer diagnostic. A
normal deployment should use `ovlab run` instead. Run one native policy
service in the foreground:

```bash
./ovlab service serve CONFIG
```

The default socket is a user-specific AF_UNIX path below
`/tmp/ovlab-<uid>/`. A diagnostic path can be supplied explicitly:

```bash
./ovlab service serve CONFIG --socket /tmp/my-policy.sock
```

Service behavior:

- no daemonization, PID file, or TCP listener;
- socket mode `0600` in a user-only directory;
- an existing socket is never overwritten;
- SIGINT and SIGTERM close the adapter and remove only the owned socket;
- service diagnostics remain attached to the foreground process.

Native development requires the policy's compatible environment. Docker
deployment does not expose this environment split to the user.

## 4. Low-level connectivity probe

With the service running, perform a lifecycle and capability handshake:

```bash
./ovlab connect CONFIG
```

`connect` reports:

- protocol version;
- policy and normalization identity;
- observation requirements;
- single-action and chunk support;
- shared `ActionSpec`;
- compatibility status and structured issues;
- scientific and execution hashes.

It does not reset an episode, request a prediction, start LIBERO, or create an
inference trace. Privileged benchmark signals are not part of the RPC schema.

## 5. Unified experiment execution

```bash
./ovlab run experiments/training/openvla-libero10-quic-smoke.yaml --plan
./ovlab run experiments/evaluation/libero10-lora-4bit-three-per-task.yaml
./ovlab run experiments/workflows/libero-model-tree.yaml
```

Standalone training and evaluation inputs become one-node trees. Every execution
preserves source, resolved plan and phase results under
`runs/<instance>/.experiment/`. Phase runs live below
`runs/<instance>/{train,evaluate}/<node-path>/<run-id>/`. Settings inherit from
parent nodes; arrays and scalar values replace inherited values. Scheduling is
root `execution.max_workers` (1–32, default 2). Configure data paths through the
existing host environment; run-directory placement does not change scientific
identity. The worker's native runner entry point is private and never recursively
starts another experiment.

`--plan` validates descriptors and dependencies without starting workloads,
loading a model, creating services or writing run/export directories. Workload
preflight checks resource availability before execution.

## 6. Verify configurations and canonical evidence

```bash
./ovlab data list runs --detail
./ovlab run CONFIG --verify
./ovlab run EXPERIMENT_ID_OR_HASH --verify
./ovlab run RUN_ID_OR_HASH --verify
./ovlab report EXPERIMENT_ID_OR_HASH
```

A configuration is resolved and validated. An existing experiment reference is
checked against its spec hash, plan and recorded phase inventory; every canonical
phase run is verified with its own artifact verifier. Individual run references
verify just that run. Run hashes are eight characters; current experiment suffix
hashes are ten. Exact IDs take precedence and ambiguous hashes fail closed.

Evaluation verification checks final manifests, scientific/execution hashes,
trace schemas, stored SHA-256 checksums, array shape and dtype, and episode counts.
Training verification checks its finalized canonical manifest and recorded file
hashes. Missing phases, changed files and invalid metadata fail. Verification is
read-only and does not recreate or publish reports.

## 7. Recompute metrics offline

```bash
./ovlab metrics recompute RUN_PATH
```

The command does not run a policy, simulator, or benchmark; does not modify the
original trace or stored metrics; records metric implementation identity;
preserves unavailable results as `status=unavailable, value=null`; and compares
complete recorded and recomputed `MetricResult` objects.

## 8. Reports and exports

```bash
./ovlab report RUN_ID
./ovlab report EXPERIMENT_ID --json
./ovlab report profiles
./ovlab report verify --run RUN_ID --profile libero-task-default --build DERIVED_BUILD_ID
```

The report command automatically creates isolated and hierarchical grouped
exports. `report publish RUN_ID` generates the HTML report and isolated export.
Explicit export commands remain available for existing automation.

Reports read canonical evidence from `OVLAB_RUNS_ROOT` and publish staged,
checksummed offline HTML/JSON builds under
`OVLAB_EXPORTS_ROOT/derived`. Exports read
canonical runs directly and publish readable CSV tables plus PNG/PDF figures
under `OVLAB_EXPORTS_ROOT`; they never scrape report HTML. `--json` returns the
final host-visible path. These commands do not start
LIBERO, a policy service, inference, or a network request.

The source-tree launcher runs report and export commands in the locked,
purpose-built `ovlab-reporting` image. It mounts canonical runs read-only and
the unified exports root read-write, while reporting real host output paths. No Conda
activation or host package installation is required. Set
`OVLAB_REPORTING_RUNTIME=host` only for an intentionally prepared native Python.

`train report` uses that same image, but mounts canonical
`OVLAB_RUNS_ROOT` read-only and writes under
`OVLAB_EXPORTS_ROOT/train/derived`. It reports exact parameter classes, PyTorch CUDA
allocator VRAM/peaks and explicitly estimated—not measured—GFLOPs.

During deployment, benchmark and policy containers are torn down after the
canonical run is finalized. The orchestrator then invokes `report publish` in
the reporting container to generate the final HTML report and isolated export.
Experiment workflows automatically produce grouped comparisons for eligible
hierarchical branches. The experiment setting is
`reporting: {enabled: true, profile: libero-task-default,
on_task_finalize: true, on_run_finalize: true, failure_policy: warn}`. Setting
`enabled: false` skips HTML but retains isolated export. A renderer or export
failure cannot change canonical benchmark status or timing. The H.1 machine
report remains available through `report publish RUN_ID --output PATH`; this
also publishes the isolated export. With the reporting container, `PATH` must
be under `OVLAB_EXPORTS_ROOT`.

## 9. Data lifecycle management

```bash
./ovlab data list
./ovlab data archive --run RUN_ID --dry-run
./ovlab data archive --all
./ovlab data delete --report RUN_ID
./ovlab data delete --export isolated:RUN_ID
./ovlab data delete --run INCOMPLETE_RUN_ID --force
./ovlab data list --archived
```

`data archive` moves selected finalized runs, completed reports, or completed
exports beneath `OVLAB_DATA_ROOT/archive` while preserving the `runs/` and
`exports/` layout. `data delete` is irreversible. Both
operations require an interactive confirmation; automation must supply
`--yes`, and should normally perform a `--dry-run` first. `--all` means all
benchmark runs, derived reports, and exports only—it never includes datasets,
checkpoints, `runs/train`, or source files. If any selected item is active or
incomplete, the complete operation is refused rather than silently skipping it.
`data archive --force` and `data delete --force` explicitly permit operations on
active or incomplete artifacts. They bypass only the state guard: confirmation,
managed-root containment and symbolic-link rejection still apply; deletion also
performs recursive permission preflight. Stop active writers before forcing an
archive.
Training report IDs use `training:TRAINING_RUN_ID`. See
[`DATA_MANAGEMENT.md`](../../../DATA_MANAGEMENT.md) for archive layout, safety
checks and concurrency guidance.

## JSON output

Commands supporting `--json` emit exactly one JSON document to stdout:

```json
{
  "schema_version": "ovlab-cli-output/1.0.0",
  "command": "policy list",
  "status": "success",
  "result": {},
  "errors": []
}
```

Logs and diagnostics go to stderr. JSON contains no ANSI sequences or Python
`repr` values. Tracebacks are hidden unless `--debug` is requested.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Success |
| `2` | Command-line usage error |
| `3` | Configuration or validation error |
| `4` | Policy/provider unavailable or incomplete |
| `5` | Service, protocol, or capability-negotiation error |
| `6` | Benchmark or runtime failure |
| `7` | Run or trace integrity failure |
| `8` | Offline metric recomputation failure |
| `130` | Interrupted by SIGINT or SIGTERM |

## QuIC descriptor-only configurations

Gate F QuIC configurations can be inspected safely:

```bash
./ovlab config validate \
  configs/policies/openvla-quic/quic-peft-bones.yaml \
  --mode descriptor

./ovlab policy describe \
  configs/policies/openvla-quic/quic-wc-bones.yaml
```

These are not runnable examples. Runtime validation, service startup, and
dry-run fail explicitly with `QuICPEFTIntegrationIncompleteError` or
`QuICWCImplementationIncompleteError`. QP0 means that no active QuIC
transformation is configured. LoRA and OpenVLA-OFT are not QP0.

## Current limitations

The CLI does not provide TCP services, daemon management, schedulers, parallel
execution, run resumption, a TUI, automatic Conda environment switching, or
QuIC runtime implementations. Conda switching is intentionally unnecessary in
the Docker deployment workflow.

Use `./ovlab --help` and the subcommand help pages as the authoritative option
reference.
