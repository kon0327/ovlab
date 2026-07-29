# OVLAB CLI

`ovlab` is the unified command-line interface for OpenVLABenchmark. It exposes
the existing configuration, policy-service, runner, artifact, and metric APIs
through one workflow:

```text
Experiment -> Deploy -> Run -> Verify -> Report
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

## Recommended Docker workflow

Copy the deployment environment template once and set the host paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Then run a complete isolated policy-service plus LIBERO benchmark topology with
one command:

```bash
./ovlab deploy run \
  configs/experiments/libero10-lora-merged-rpc-smoke.yaml \
  --profile openvla \
  --renderer egl
```

The CLI validates the Compose model, passes the same experiment to both
containers, waits for policy readiness, runs the benchmark, propagates failure,
and removes its containers and private RPC volume. Canonical artifacts remain in
the host directory configured by `OVLAB_RUNS_ROOT`.

LIBERO datasets follow the same external-data convention. If
`OVLAB_DATASETS_PATH` is unset, deployment derives
`<OVLAB_RUNS_ROOT>/../datasets/libero`, creates the directory, and mounts it
read-only. An explicitly configured dataset path must already exist.

Use `--profile oft` for an OpenVLA-OFT experiment. Use `--renderer glfw` only
with an interactive display server. Preview the exact commands without touching
Docker:

```bash
./ovlab deploy run CONFIG --profile openvla --renderer egl --dry-run
```

Before Docker starts, the CLI resolves the policy's portable `checkpoint_id`.
It checks `~/.cache/huggingface`, an optional local-profile `local_path`, and
`ovlab-data/checkpoints/huggingface`; if still absent, it downloads exactly the
registry-pinned revision into managed storage. File sizes and SHA-256 values are
verified before the snapshot is mounted read-only. Resolution, per-file download
progress, transferred byte counts, and verification stages are printed to
stderr; `--json` stdout therefore remains machine-readable. Use `--offline` to
prohibit the download path:

```bash
./ovlab deploy run CONFIG --profile oft --offline
```

Custom unpublished checkpoints remain outside the repository and are selected
through a gitignored local profile:

```bash
./ovlab deploy run CONFIG --profile openvla \
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
│   ├── validate CONFIG [--mode descriptor|runtime] [--json]
│   └── resolve CONFIG [--mode descriptor|runtime] [--format yaml|json]
├── policy
│   ├── list [--json]
│   └── describe CONFIG [--json]
├── service
│   ├── serve CONFIG [--socket PATH]
│   └── health --socket PATH [--json]
├── connect CONFIG [--json]
├── deploy
│   └── run EXPERIMENT --profile openvla|oft [--renderer egl|glfw]
│       [--env-file PATH] [--local-profile PATH] [--offline]
│       [--project-name NAME] [--dry-run] [--json]
├── run CONFIG [--output-root PATH] [--dry-run] [--json]
├── run inspect RUN_PATH [--json]
├── run verify RUN_PATH [--json]
├── metrics recompute RUN_PATH [--json]
└── report generate RUN_PATH --output PATH [--json]
```

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

## 8. Generate a derived report

```bash
./ovlab report generate RUN_PATH --output DERIVED_PATH
```

The source run remains immutable. In Docker deployment, the reporting profile
mounts canonical runs read-only and writes only to the configured derived root.

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
