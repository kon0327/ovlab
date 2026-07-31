# OpenVLABenchmark release notes

## 0.2.0 — 2026-07-31

OVLAB 0.2.0 is the first consolidated experimental-framework release. It
summarizes development from the initial repository skeleton on 2026-07-15
through the dataset and training pipeline, together with the current CLI
usability work. The repository has no earlier release tag; the historical
development baseline used package and CLI version `0.1.0`.

### Framework and reproducibility foundations

- Established the monorepo layout for core contracts, benchmark and policy
  adapters, the experiment runner, metrics, CLI, deployment, tests, laboratory
  analysis, configurations and pinned external Git submodules.
- Added strict YAML composition with explicit inheritance, kind-specific schema
  validation and separate scientific and execution configuration hashes.
- Defined typed lifecycle, observation, action, signal, capability, identifier,
  timing, error and compatibility contracts.
- Added deterministic task, episode, rollout, seed and identifier handling,
  including requested-versus-applied actions and supported action-chunk modes.
- Implemented immutable filesystem artifacts, finalized trace checksums, reload
  verification and identical offline metric recomputation.

### LIBERO benchmark integration

- Added the LIBERO benchmark adapter with deterministic task and initial-state
  selection, canonical RGB observations, authoritative instructions and
  environment-owned success semantics.
- Added explicit EGL and GLFW execution profiles. Renderer selection is applied
  before graphics-library import, contributes only to the execution hash and is
  recorded in provenance.
- Added portable LIBERO smoke, episode, multi-task and qualification experiment
  configurations.
- Added per-episode metadata and default inference-video capture. Videos use
  web-compatible H.264/AVC, `yuv420p` and fast-start metadata.
- Adopted readable run IDs in the form
  `experiment-name_YYYY-MM-DD_HH-MM-SS_shorthash`.

### Isolated policy services

- Added a generic `RemotePolicyAdapter` and versioned, length-prefixed AF_UNIX
  protocol with bounded lifecycle, health, prediction and cleanup operations.
- Restricted policy-visible RPC input to identifiers, authoritative instruction
  and canonical RGB bytes with explicit shape, layout and dtype metadata.
- Recorded service-side synchronized inference time, runner-observed RPC
  round-trip time and closed-loop step time independently.
- Integrated OpenVLA Vanilla, merged OpenVLA-LoRA and OpenVLA-OFT policy
  services while keeping their dependency stacks isolated from the runner.
- Added 4-bit inference support for Vanilla and merged-LoRA configurations,
  with explicit method and quantization provenance.
- Added QuIC-PEFT and QuIC-WC contract wrappers, source-intake validation and a
  dedicated pinned OpenVLA-QuIC external source. These wrappers remain
  experimental and are not presented as a completed benchmark evaluation.

### Unified CLI and configuration workflow

- Added the checkout-local `./ovlab` entry point for configuration, policy
  service, connection, execution, deployment, metrics, reports, exports,
  datasets, training and checkpoints.
- Added Docker Compose orchestration that resolves the experiment profile and
  renderer from portable YAML while retaining explicit CLI overrides.
- Added dynamic read-only configuration bundles so new experiment YAML files do
  not require an image rebuild.
- Added concise operator-oriented output by default, `--detail` for complete
  interactive documents and `--json` for the stable machine-readable envelope.
- Added categorized exit codes, actionable errors, progress reporting and
  foreground service-log propagation.

### Reproducible deployment and model artifacts

- Added purpose-built, digest-pinned benchmark, OpenVLA, OpenVLA-OFT,
  reporting, dataset and training images with dependency locks and source
  manifests.
- Added generic checkpoint resolution by logical ID and pinned revision. The
  resolver reuses verified global Hugging Face snapshots, supports host-local
  artifacts and maintains an OVLAB-managed cache outside the source repository.
- Mounted only the resolved checkpoint snapshot into policy containers at a
  stable read-only path and retained host location as execution provenance only.
- Established the external `ovlab-data` workspace for checkpoints, datasets,
  immutable runs, derived reports and publication exports.
- Added automatic checkpoint verification and visible acquisition progress
  before container startup.

### Reporting and exports

- Added a dedicated offline reporting engine that mounts canonical runs
  read-only and writes regenerable content to `derived/`.
- Added self-contained interactive HTML/SVG charts with labeled axes, cursor
  values, zoom, pan and descriptive-statistics tables.
- Added integrity manifests and deterministic report build IDs.
- Added readable isolated exports at run and episode scope and explicitly
  requested grouped exports across all, same-model or manually selected runs.
- Added statistical tables, success comparisons, action time series,
  distributions and trajectory-oriented figures without altering source runs.

### Dataset and training pipeline

- Added explicit dataset discovery, resolution, acquisition, local import,
  preparation, inspection and checksum verification in a dedicated container.
- Added immutable readable dataset locations under
  `datasets/<provider>/<name>/<version>` while preserving complete content and
  source identity in manifests and retaining legacy-layout readability.
- Added strict, portable training profiles, offline planning, resource checks,
  isolated OpenVLA LoRA training and atomic checkpoint finalization.
- Kept dataset acquisition explicit, disabled trainer network access and
  separated staging output from the finalized checkpoint registry.
- Added training-run and checkpoint status, inspection and integrity commands.

### Testing and documentation

- Added dependency-light unit, contract and integration suites for core
  contracts, configuration, adapters, RPC, runner, metrics, artifacts, CLI,
  deployment, reporting, datasets and training.
- Added marked real LIBERO/GPU/manual paths without allowing skipped hardware
  tests to stand in for real qualification.
- Added operator guides for the CLI, Docker deployment, reporting and export,
  datasets, training, manual policy-service smoke tests and the analysis lab.

### Compatibility and migration notes

- Product, CLI, Python distribution, runtime component and OCI image versions
  advance together from `0.1.0` to `0.2.0`.
- The strict configuration schema and core compatibility contract remain
  `0.1.0`; the policy RPC protocol remains `ovlab-policy-rpc/1.0.0`; the CLI
  JSON envelope remains `ovlab-cli-output/1.0.0`. Existing conforming configs,
  traces and automation therefore do not require a schema migration.
- Existing datasets in the former source-revision/build-ID directory layout
  remain discoverable. New publications use the readable version layout.
- Existing Docker images contain the previous packaged source and version
  labels. Rebuild the roles used by an operator before running 0.2.0.
- External repositories, model checkpoints, benchmark datasets, canonical
  runs and derived data are not modified by this version bump.

### Known scope limits

- OVLAB remains an experimental research framework rather than a stable 1.0
  public API.
- QuIC wrappers and training support are intentionally narrower than the
  validated Vanilla, merged-LoRA and OFT execution paths.
- Interactive GLFW rendering requires an available display stack; automated
  and headless LIBERO execution should use EGL.
- Full benchmark campaigns and model training still require suitable local GPU,
  checkpoint and dataset resources.
