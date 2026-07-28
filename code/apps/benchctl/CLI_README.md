# OVLAB CLI

`ovlab` is the unified command-line interface for OpenVLABenchmark. It exposes
the existing configuration, policy-service, runner, artifact, and metric APIs
through one workflow:

```text
Config -> Connect -> Run -> Inspect
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

The source-tree launcher does not install or update packages.

## Machine-local configuration

Runtime commands require a local profile containing host-specific paths and
device selection:

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
│   └── serve CONFIG [--socket PATH]
├── connect CONFIG [--json]
├── run CONFIG [--output-root PATH] [--dry-run] [--json]
├── run inspect RUN_PATH [--json]
├── run verify RUN_PATH [--json]
└── metrics recompute RUN_PATH [--json]
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

## 3. Start a policy service

Run one policy service in the foreground:

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

Run the service in the policy's compatible environment. The runner and an
OpenVLA policy service may require different existing Conda environments.

## 4. Probe connectivity

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

## 5. Plan or execute a run

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

Execute through the existing `ExperimentRunner`:

```bash
./ovlab run CONFIG
```

Use a machine-local output placement when needed:

```bash
./ovlab run CONFIG --output-root /path/to/runs
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
execution, run resumption, a TUI, Docker/Compose entrypoints, automatic Conda
environment switching, or QuIC runtime implementations.

Use `./ovlab --help` and the subcommand help pages as the authoritative option
reference.
