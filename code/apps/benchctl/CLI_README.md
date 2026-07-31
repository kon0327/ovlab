# OVLAB CLI

`ovlab` is the unified command-line interface for OpenVLABenchmark. It exposes
the existing configuration, policy-service, runner, artifact, and metric APIs
through one workflow:

```text
Dataset -> Train -> Checkpoint -> Deploy -> Run -> Verify -> Report
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

## Output modes

Structured commands use one consistent output contract:

- no output flag: compact operator-oriented rows or `key  value` summaries;
- `--detail`: the complete result document as readable indented JSON;
- `--json`: one stable `ovlab-cli-output/1.0.0` JSON envelope for scripts.

`--detail` and `--json` are mutually exclusive. Operational progress and
diagnostics remain on stderr, so `--json` stdout stays machine-readable. The
`config resolve` command is intentionally different: its result is the complete
resolved configuration and `--format yaml|json` selects its serialization.

## Recommended Docker workflow

Copy the deployment environment template once and set the host paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Then run a complete isolated policy-service plus LIBERO benchmark topology with
one command:

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml
```

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

`--profile` and `--renderer` are optional overrides with precedence over the
experiment values. A profile override must remain compatible with the selected
policy type. Use GLFW only with an interactive display server. Preview the
resolved selection and exact commands without touching Docker:

```bash
./ovlab deploy run CONFIG --dry-run
./ovlab deploy run CONFIG --renderer glfw --dry-run
```

Run the five-task cross-suite Vanilla NF4 experiment with:

```bash
./ovlab deploy run \
  configs/experiments/libero10-openvla-vanilla-4bit-five-episodes.yaml
```

The selected policy component owns `runtime.quantization: none | 4bit`; there is
no host-side quantization switch. This keeps the inference representation in the
portable scientific configuration and its hash.

Before Docker starts, the CLI resolves the policy's portable `checkpoint_id`.
It checks `~/.cache/huggingface`, an optional local-profile `local_path`, and
`ovlab-data/checkpoints/huggingface`; if still absent, it downloads exactly the
registry-pinned revision into managed storage. File sizes and SHA-256 values are
verified before the snapshot is mounted read-only. Resolution, per-file download
progress, transferred byte counts, and verification stages are printed to
stderr; `--json` stdout therefore remains machine-readable. Use `--offline` to
prohibit the download path:

```bash
./ovlab deploy run CONFIG --offline
```

Custom unpublished checkpoints remain outside the repository and are selected
through a gitignored local profile:

```bash
./ovlab deploy run CONFIG \
  --local-profile configs/local/profile.yaml
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

## Command overview

```text
ovlab
├── --help
├── --version
├── config
│   ├── validate CONFIG [--mode descriptor|runtime] [--detail | --json]
│   └── resolve CONFIG [--mode descriptor|runtime] [--format yaml|json]
├── policy
│   ├── list [--detail | --json]
│   └── describe CONFIG [--detail | --json]
├── service
│   ├── serve CONFIG [--socket PATH] [--detail | --json]
│   └── health --socket PATH [--detail | --json]
├── connect CONFIG [--detail | --json]
├── deploy
│   └── run EXPERIMENT [--profile openvla|oft] [--renderer egl|glfw]
│       [--env-file PATH] [--local-profile PATH] [--offline]
│       [--project-name NAME] [--dry-run] [--detail | --json]
├── run CONFIG [--output-root PATH] [--dry-run] [--detail | --json]
├── run inspect RUN_PATH [--detail | --json]
├── run verify RUN_PATH [--detail | --json]
├── metrics recompute RUN_PATH [--detail | --json]
├── report profiles [--detail | --json]
├── report generate --run RUN_ID [--task TASK_ID] [--profile PROFILE] [--detail | --json]
├── report publish --run RUN_ID [--profile PROFILE] [--report-enabled true|false] [--detail | --json]
├── report verify --run RUN_ID [--profile PROFILE] [--build BUILD_ID] [--detail | --json]
├── export isolated --run RUN_ID [--episode EPISODE_ID] [--template TEMPLATE] [--detail | --json]
├── export grouped --name GROUP (--all-runs | --same-model-as RUN_ID | --runs RUN_ID...)
│   [--suite SUITE] [--template TEMPLATE] [--detail | --json]
├── export verify --kind isolated|grouped --name NAME [--detail | --json]
├── export generate --spec SPEC.yaml [--detail | --json]  # legacy grouped bridge
├── dataset
│   ├── providers [--detail | --json]
│   ├── resolve --benchmark libero --suite SUITE [--detail | --json]
│   ├── fetch --source libero|url --name NAME [URL OPTIONS] [--detail | --json]
│   ├── import --name NAME --version VERSION --path PATH [--detail | --json]
│   ├── prepare --dataset DATASET_ID --format FORMAT [--detail | --json]
│   ├── list [--detail | --json]
│   ├── inspect --dataset DATASET_ID [--detail | --json]
│   └── verify --dataset DATASET_ID [--detail | --json]
├── train
│   ├── profiles [--detail | --json]
│   ├── validate --profile PROFILE [--detail | --json]
│   ├── plan --profile PROFILE [--detail | --json]
│   ├── run --profile PROFILE [--allow-dataset-download] [--detail | --json]
│   ├── status --run TRAINING_RUN_ID [--detail | --json]
│   ├── inspect --run TRAINING_RUN_ID [--detail | --json]
│   └── verify --run TRAINING_RUN_ID [--detail | --json]
└── checkpoint
    ├── list [--detail | --json]
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
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
  --mode descriptor
```

Runtime validation additionally requires a complete runnable configuration:

```bash
./ovlab config validate \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
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

Describe the policy selected by an experiment:

```bash
./ovlab policy describe CONFIG
```

The description reports the method family, configured artifact, capabilities,
readiness, and available scientific and execution identities. It does not load
a model or initialize CUDA.

## 3. Low-level native policy service

This command is an internal container entrypoint and a developer diagnostic. A
normal deployment should use `ovlab deploy run` instead. Run one native policy
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

## 5. Low-level native runner

Inspect the execution plan without runtime side effects:

```bash
./ovlab run CONFIG --dry-run
```

Dry-run guarantees:

- complete resolution and runtime validation;
- no model or checkpoint loading;
- no CUDA initialization;
- no service or socket creation;
- no prediction or benchmark rollout;
- no run-directory creation.

Inside the benchmark container, execute through the existing
`ExperimentRunner`:

```bash
./ovlab run CONFIG
```

Use a machine-local output placement when needed:

```bash
./ovlab run CONFIG --output-root /var/lib/ovlab/runs
```

Changing output placement does not change scientific identity. Existing run
directories are never overwritten. The CLI does not provide an implicit
fallback to Vanilla, LoRA, OFT, mock policies, or dummy actions.

## 6. Inspect and verify runs

Read a run summary without changing artifacts:

```bash
./ovlab run inspect RUN_PATH
```

Verify trace and artifact integrity:

```bash
./ovlab run verify RUN_PATH
```

Verification checks required artifacts, final manifest consistency,
scientific/execution hashes, trace schemas and finalization markers, referenced
array shape and dtype, stored SHA-256 checksums, and episode counts. It does not
claim guarantees beyond the hashes stored by OVLAB.

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
./ovlab report profiles
./ovlab report generate --run RUN_ID --profile libero-task-default
./ovlab report generate --run RUN_ID --task libero/10/0 --profile libero-task-default
./ovlab report verify --run RUN_ID --profile libero-task-default --build DERIVED_BUILD_ID
./ovlab export isolated --run RUN_ID
./ovlab export isolated --run RUN_ID --episode EPISODE_ID
./ovlab export grouped --name paper-ablation --runs RUN_ID_A RUN_ID_B
./ovlab export grouped --name model-family --same-model-as REFERENCE_RUN_ID
./ovlab export grouped --name complete-study --all-runs --suite libero_10
./ovlab export verify --kind grouped --name paper-ablation
```

Reports read canonical evidence from `OVLAB_RUNS_ROOT` and publish staged,
checksummed offline HTML/JSON builds under `OVLAB_DERIVED_ROOT`. Exports read
canonical runs directly and publish readable CSV tables plus PNG/PDF figures
under `OVLAB_EXPORTS_ROOT`; they never scrape report HTML. `--json` returns the
final host-visible path. These commands do not start
LIBERO, a policy service, inference, or a network request.

The source-tree launcher runs report and export commands in the locked,
purpose-built `ovlab-reporting` image. It mounts canonical runs read-only,
derived/exports read-write, and still reports real host output paths. No Conda
activation or host package installation is required. Set
`OVLAB_REPORTING_RUNTIME=host` only for an intentionally prepared native Python.

During deployment, benchmark and policy containers are torn down after the
canonical run is finalized. The orchestrator then invokes `report publish` in
the reporting container to generate the final HTML report and isolated export.
Grouped comparison is never automatic. The experiment setting is
`reporting: {enabled: true, profile: libero-task-default,
on_task_finalize: true, on_run_finalize: true, failure_policy: warn}`. Setting
`enabled: false` skips HTML but retains isolated export. A renderer or export
failure cannot change canonical benchmark status or timing. The legacy `report generate RUN_PATH
--output PATH` form remains available for the H.1 machine report.

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
