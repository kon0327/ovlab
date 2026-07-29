# ovlab-benchctl

This package is OVLAB's configuration composition root. It deliberately sits
above concrete benchmark, policy, metric, runner, and artifact packages. Those
owner packages remain independently importable and do not depend on this
resolver.

Configuration uses a strict dependency-free YAML subset, explicit relative
`extends`, exact per-kind schemas, and explicit experiment component
references. Duplicate keys, unknown keys, implicit merge keys, anchors,
aliases, tags, multi-document files, non-two-space indentation, reference
traversal, and inheritance cycles are rejected. Mapping inheritance is a deep
merge; sequences and scalars replace their parent value.

`ConfigResolver.resolve()` validates and composes all component documents,
resolves logical checkpoint/device/artifact resources through an explicitly
selected local profile, composes a separately selected renderer execution
profile for LIBERO, constructs the existing immutable owner settings, and
checks the shared action and observation interfaces. It returns separate
scientific and execution hashes. The scientific hash excludes the local and
renderer profiles plus resolved machine paths/devices; the execution hash
includes them. `profiles/libero-bench-egl.yaml` is the default LIBERO execution
profile; callers may explicitly select `profiles/libero-playground-glfw.yaml`.
Diagnostic `MUJOCO_GL` and `MUJOCO_EGL_DEVICE_ID` values override the selected
profile and are reflected in the resolved execution configuration.

The host-side deployment resolver reads only the selected policy's logical
checkpoint ID and the portable registry identity. It checks the default global
Hugging Face cache, an optional local-profile override, and OVLAB-managed
storage, verifies declared file sizes and SHA-256 values, and exposes only the
resolved snapshot to the policy container. Missing artifacts are downloaded at
their pinned revision unless `ovlab deploy run --offline` is used. This path uses
the Python standard library and does not require a host Conda environment.

`ResolvedExperimentConfig.write()` creates one deterministic
`resolved_config.yaml` and refuses to overwrite an existing file.

## Unified CLI

See [CLI_README.md](CLI_README.md) for the complete command reference and
workflow guide.

Gate G exposes the composition root as one `ovlab` command. An installed
`ovlab-benchctl` package provides the `ovlab` console script. From a source
checkout, `./ovlab` is the installation-independent invocation and assembles
the repository's existing src-layout packages without installing anything.

The intended workflow is Config -> Connect -> Run -> Inspect:

```bash
export OVLAB_LOCAL_PROFILE=configs/local/profile.yaml

./ovlab config validate configs/experiments/libero10-lora-merged-rpc-smoke.yaml --mode runtime
./ovlab policy describe configs/experiments/libero10-lora-merged-rpc-smoke.yaml
./ovlab service serve configs/experiments/libero10-lora-merged-rpc-smoke.yaml
./ovlab connect configs/experiments/libero10-lora-merged-rpc-smoke.yaml
./ovlab run configs/experiments/libero10-lora-merged-rpc-smoke.yaml --dry-run
./ovlab run inspect RUN_PATH
./ovlab run verify RUN_PATH
./ovlab metrics recompute RUN_PATH
```

`OVLAB_LOCAL_PROFILE` selects the machine-local resource profile; when it is
unset, `configs/local/profile.yaml` is used. `OVLAB_EXECUTION_PROFILE` may
select a renderer execution profile. The local AF_UNIX socket defaults to a
user-specific path below `/tmp/ovlab-<uid>/`; `service serve --socket` is
available for diagnostics. These placement choices do not change scientific
identity.

`config validate` and `config resolve` accept `--mode descriptor` for safe
inspection and `--mode runtime` for runnable-resource validation. QuIC-PEFT
and QuIC-WC Gate F files are descriptor-only: the following commands are safe
and do not imply runnable implementations:

```bash
./ovlab config validate configs/policies/openvla-quic/quic-peft-bones.yaml --mode descriptor
./ovlab policy describe configs/policies/openvla-quic/quic-wc-bones.yaml
```

Runtime validation, service startup, and dry-run intentionally reject those
skeletons with their typed Gate I/Gate J incomplete-implementation errors.
They are not runnable examples. QP0 in their descriptors means that no active
QuIC transformation is configured; LoRA and OpenVLA-OFT remain independent
methodological references.

Human-readable output is the default. Commands with `--json` write exactly one
versioned JSON document to stdout and route diagnostics to stderr. Stable exit
codes are:

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 2 | Command-line usage error |
| 3 | Configuration or validation error |
| 4 | Policy/provider unavailable or incomplete |
| 5 | Service, protocol, or capability error |
| 6 | Benchmark/runtime failure |
| 7 | Run/trace integrity failure |
| 8 | Offline metric recomputation failure |
| 130 | Interrupted by SIGINT or SIGTERM |

`service serve` remains in the foreground and never creates a daemon, PID
file, or TCP listener. SIGINT/SIGTERM closes the adapter and removes only the
owned socket. `connect` performs initialize/health/capability negotiation and
then closes without prediction or trace creation. `run` delegates execution to
`ExperimentRunner`; it contains no rollout loop. `--dry-run` resolves and
runtime-validates the full configuration but loads no model, initializes no
CUDA context, opens no socket, and creates no run directory.

`run inspect` and `run verify` are read-only. Verification uses the stored
array SHA-256 values and schema/finalization invariants without claiming any
stronger cryptographic guarantee. `metrics recompute` invokes the offline
metric API, compares complete recorded and recomputed episode results, and
never modifies or replaces the original trace or stored result.

Gate G does not provide daemon management, TCP, scheduling, resumption,
parallel runs, shell completion, a TUI, Docker packaging, QuIC algorithms, or
QuIC runtime readiness. A real service must be launched in its compatible
existing policy environment, while the LIBERO runner remains in its compatible
runner environment.
