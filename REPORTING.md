# OVLAB Reporting and Export Operations

This guide explains how to generate, inspect, verify, and export OVLAB reports.
Reporting is an offline postprocessing workflow: it reads canonical benchmark
evidence from `runs/`, writes regenerable reports to `derived/`, and writes
explicit publication exports to `exports/`.

Reporting never starts LIBERO, loads a policy checkpoint, or performs model
inference. Canonical run artifacts remain authoritative and read-only.

## Storage contract

The recommended host workspace is outside the source repository:

```text
/home/kony/dissertation/ovlab-data/
├── runs/       # immutable traces, metrics, videos, configurations and provenance
├── derived/    # regenerable benchmark and training HTML reports
└── exports/    # explicitly selected figures and tables
```

The CLI uses the repository sibling `ovlab-data/` by default. Override the
locations only when required by the host:

```bash
export OVLAB_DATA_ROOT=/home/kony/dissertation/ovlab-data
export OVLAB_RUNS_ROOT="$OVLAB_DATA_ROOT/runs"
export OVLAB_DERIVED_ROOT="$OVLAB_DATA_ROOT/derived"
export OVLAB_EXPORTS_ROOT="$OVLAB_DATA_ROOT/exports"
```

Host paths, creation timestamps, renderer devices, and Docker tags do not enter
the scientific configuration hash.

## Automatic postprocessing after a benchmark

Reporting is enabled by default. A portable experiment may state the behavior
explicitly:

```yaml
reporting:
  enabled: true
  profile: libero-task-default
  on_task_finalize: true
  on_run_finalize: true
  failure_policy: warn
```

In Docker deployment, the benchmark process only finalizes the canonical run.
After benchmark and policy teardown, the orchestrator starts the dedicated
`ovlab-reporting` image and hands it the single new run ID. That container mounts
`runs/` read-only and atomically publishes both the complete HTML report under
`derived/` and the isolated run export under `exports/`.

`reporting.enabled: false` skips the HTML report but retains the automatic
isolated export. The historical task-finalization fields remain accepted for
native runner compatibility; the external container handoff publishes only
after canonical run finalization. A publication failure is reported separately
with the completed canonical run path and cannot change its status or contents.

Use the normal deployment command; no separate reporting command is required:

```bash
./ovlab deploy run configs/experiments/EXPERIMENT.yaml
```

Reporting configuration affects only derived identity; it does not affect
scientific or execution configuration hashes.

## Standalone report generation

List installed report profiles:

```bash
./ovlab report profiles
./ovlab report profiles --json
```

Generate a complete run report from an existing canonical run:

```bash
RUN_ID=experiment-name_2026-07-30_15-18-22_fd24dff0
# Equivalent short reference shown by `./ovlab data list --kind runs`:
RUN_HASH=fd24dff0

./ovlab report generate \
  --run "$RUN_ID" \
  --profile libero-task-default
```

All benchmark report and export selectors accept either `RUN_ID` or the
displayed eight-character `RUN_HASH`. This includes grouped `--runs` and
`--same-model-as`, isolated export verification, and report verification.
Exact IDs win; ambiguous short hashes are rejected rather than guessed.

Generate a task-scoped report when investigating one task:

```bash
./ovlab report generate \
  --run "$RUN_ID" \
  --task libero-10-0-2ef7121ebcbf \
  --profile libero-task-default
```

The command returns the report directory and deterministic
`derived_build_id`. Add `--json` for automation:

```bash
./ovlab report generate \
  --run "$RUN_ID" \
  --profile libero-task-default \
  --json
```

An unchanged source run, profile, template bundle, and renderer version reuse
the same build instead of creating a semantically duplicate report.

## Report output layout

Reports are stored by canonical run ID, profile, and deterministic build ID:

```text
derived/<run-id>/<profile-id>/
├── latest.json
└── <derived-build-id>/
    ├── index.html
    ├── report.json
    ├── manifest.json
    ├── assets/
    │   ├── style.css
    │   └── charts/*.svg
    └── tasks/<task-key>/index.html
```

Open `index.html` directly in a browser. Reports are self-contained and do not
use a CDN, external font, analytics service, or remote JavaScript. Episode video
links are relative links to the canonical MP4 files under the sibling `runs/`
tree, so keep `runs/` and `derived/` under the same data root when moving them.

Charts are self-contained interactive SVG documents with labeled X/Y axes and
units. Hover or move the pointer over the plot for the nearest displayed sample
and exact values, use the mouse wheel for horizontal zoom, drag to pan, and
double-click to restore the full range. The action chart shows all seven
canonical applied-action components. The SVG remains readable as a static chart
when scripting is disabled.

Each chart is followed by a descriptive-statistics table computed from the full
canonical series, not from the downsampled display points. Tables record task and
episode scope, series, unit, finite `n`, non-finite count, minimum, P05, median,
mean, sample standard deviation, P95, and maximum. Sample standard deviation is
reported as unavailable for `n < 2`; the exact population and quantile semantics
are preserved in `report.json`.

Task pages also expand canonical action and system metric aggregates into
statistical columns (valid/excluded/unavailable counts, minimum, median, mean,
sample standard deviation and maximum). Aggregate mappings are retained in
`report.json` for machine use but are not printed as JSON/Python literals in the
HTML report.

The run outcome section reports both success rate and its complete complement as
`Non-success`. It also lists canonical terminal-state counts and one row per
episode with task, rollout, seed, terminal state, classification and reason. A
`time_limit` is therefore visible as scientific non-success without being
mislabelled as a policy, benchmark or infrastructure failure.

`latest.json` identifies the most recently generated run- or task-scoped build.
For archival automation, retain the returned build ID and pass it explicitly to
verification.

## Verify a report

Verify the latest build for a run/profile:

```bash
./ovlab report verify \
  --run "$RUN_ID" \
  --profile libero-task-default
```

Verify an exact immutable derived build:

```bash
./ovlab report verify \
  --run "$RUN_ID" \
  --profile libero-task-default \
  --build DERIVED_BUILD_ID \
  --json
```

Verification checks:

- the canonical run or task checksums used to build the report;
- the deterministic build identity;
- the report manifest payload checksum;
- every generated HTML, JSON, CSS, and SVG checksum and size;
- missing or unexpected files;
- internal asset and canonical-video links;
- absence of external URLs in generated HTML.

Any mismatch exits with code `7`. Do not repair a failed verification by editing
the derived build. Remove or archive the damaged derived build and regenerate it
from the still-valid canonical run.

## Report contents and interpretation

The built-in `libero-task-default` profile includes:

- experiment, policy, benchmark, run status, and configuration hashes;
- task instruction, episode seed, initial-state identity, and terminal status;
- task, action, and system metrics with IDs, versions, units, sample counts, and
  unavailable reasons;
- applied-action trajectories and gripper transitions;
- policy-service inference latency derived from canonical prediction timing;
- policy-process PyTorch allocator VRAM tracking (`allocated`, `reserved`, and
  scoped peaks) with an interactive time-series chart;
- exact live runtime parameter counts, including separate active adapter counts
  where an unmerged adapter exists;
- estimated GFLOPs per prediction with estimator ID, formula inputs and an
  explicit warning that the value is not a hardware measurement;
- binary episode outcomes, explicit multi-episode denominators, and missing or
  interrupted counts;
- links to canonical episode videos;
- renderer and source provenance.

A one-episode task is reported as `success=true|false`, not as a statistical
success rate. Multi-episode success rates expose both numerator and denominator.
An unavailable metric remains `status=unavailable, value=null`; reporting does
not substitute zero. Standard deviation is unavailable for `n < 2`.

Merged LoRA checkpoints have no live adapter tensors, so the active adapter
count is zero or unavailable; their historical LoRA configuration remains in
method provenance and is not misreported as live runtime parameters. Vanilla,
merged LoRA, 4-bit and OFT services all emit the same telemetry schema.

## Training performance reports

Finalized training runs use the same isolated reporting image and a distinct
read-only source mount:

```bash
./ovlab train report --run TRAINING_RUN_ID
./ovlab train report --run TRAINING_RUN_ID --verify
./ovlab train report --run TRAINING_RUN_ID --verify --build BUILD_ID --json
```

`ovlab train run` generates this performance report automatically after the
checkpoint finalizer succeeds. The commands above regenerate or verify it
without starting a trainer.

Output is stored under
`derived/training/<training-run-id>/system-performance/<build-id>/`. Training
reports show optimizer-step VRAM tracking and peaks, exact total/trainable/
frozen/adapter counts, step duration and loss statistics, and estimated GFLOPs
per optimizer step and in total. The estimator is the documented dense-model
proxy `(2 * total parameters + 4 * trainable parameters) * non-padding tokens`;
it deliberately does not pretend
to measure Tensor Core utilization, optimizer work, vision preprocessing or
gradient-checkpoint recomputation. See [`TRAINING.md`](TRAINING.md) for the
canonical training evidence contract.

## Report profiles and templates

The built-in profile uses schema `ovlab.report-profile/v1`. Its packaged source
is:

```text
code/apps/runner/src/ovlab_runner/report_assets/
├── profiles/libero-task-default.yaml
├── templates/benchmark/run-v1.html
├── templates/benchmark/task-v1.html
└── static/style.css
```

A local profile may be passed as a YAML path:

```bash
./ovlab report generate \
  --run "$RUN_ID" \
  --profile configs/local/my-report-profile.yaml
```

Local profile and template paths are resolved beneath the profile directory.
Path traversal, unsupported builders, unknown fields, executable YAML, and
remote templates are rejected. Keep reusable portable profiles in the
repository and machine-specific experiments in the gitignored `configs/local/`
tree.

## Isolated and grouped exports

Exports are intentionally simpler than derived HTML reports. They contain
ordinary CSV tables, PNG/PDF figures, and one readable `metadata.json`. They
never scrape report HTML and never modify their canonical run sources.

### Isolated export

A successful benchmark run automatically receives a complete isolated export.
It can also be regenerated explicitly:

```bash
./ovlab export isolated --run RUN_ID
```

To export exactly one episode instead of the full run:

```bash
./ovlab export isolated --run RUN_ID --episode EPISODE_ID
```

The layout is:

```text
exports/isolated/<run-name>/
├── episodes/
│   ├── tables/
│   │   ├── <episode>__statistics.csv
│   │   ├── <episode>__timeseries.csv
│   │   └── <episode>__action-metrics.csv
│   └── figures/
│       ├── <episode>__actions-over-time.{png,pdf}
│       ├── <episode>__action-boxplots.{png,pdf}
│       ├── <episode>__action-metrics.{png,pdf}
│       ├── <episode>__timing-over-time.{png,pdf}
│       └── <episode>__eef-trajectory-3d.{png,pdf}  # when recorded
├── overview/                         # complete-run exports only
│   ├── tables/
│   │   ├── episode-summary.csv
│   │   ├── descriptive-statistics.csv
│   │   ├── metric-summary.csv
│   │   ├── action-metrics-by-episode.csv
│   │   └── action-metrics-summary.csv
│   └── figures/
│       ├── success-by-task.{png,pdf}
│       ├── action-boxplots.{png,pdf}
│       ├── action-metrics-by-episode.{png,pdf}
│       ├── action-metrics-by-task.{png,pdf}
│       ├── inference-latency-boxplots.{png,pdf}
│       ├── episode-lengths.{png,pdf}
│       └── eef-trajectories-3d.{png,pdf}           # when recorded
└── metadata.json
```

An episode-only request publishes a self-contained isolated directory with just
that episode and no `overview/`. A subsequent run-level request atomically
replaces it with the complete run export.

### Grouped export

Grouped comparisons are generated only on explicit request. Choose exactly one
selection mode:

```bash
# Manually selected run names
./ovlab export grouped \
  --name paper-ablation \
  --runs RUN_ID_A RUN_ID_B RUN_ID_C

# Every completed run with the same model and checkpoint as a reference run
./ovlab export grouped \
  --name openvla-family \
  --same-model-as REFERENCE_RUN_ID

# Every compatible completed run, optionally restricted to a suite
./ovlab export grouped \
  --name libero10-study \
  --all-runs \
  --suite libero_10
```

The layout is:

```text
exports/grouped/<group-name>/
├── tables/
│   ├── run-summary.csv
│   ├── episode-summary.csv
│   ├── descriptive-statistics.csv
│   ├── metric-summary.csv
│   ├── action-metrics-by-episode.csv
│   └── action-metrics-summary.csv
├── figures/
│   ├── success-comparison.{png,pdf}
│   ├── task-success-heatmap.{png,pdf}
│   ├── action-metrics-by-run.{png,pdf}
│   ├── action-metrics-by-model.{png,pdf}
│   ├── inference-latency-boxplots.{png,pdf}
│   ├── inference-latency-ecdf.{png,pdf}
│   ├── success-latency-pareto.{png,pdf}
│   ├── terminal-outcome-composition.{png,pdf}
│   └── eef-trajectories-3d.{png,pdf}                # when recorded
└── metadata.json
```

The success comparison includes 95% Wilson intervals, which remain meaningful
for small episode counts. The heatmap exposes task-specific regressions that an
aggregate success rate can hide. ECDFs preserve latency-tail behavior, and the
success/latency plot exposes deployment trade-offs. Terminal composition keeps
timeouts and benchmark or policy failures separate. Three-dimensional
end-effector trajectories are generated only when canonical position signals
were recorded; absence is declared in `metadata.json`, never fabricated.

Action-metric exports prioritize the canonical `action.variance`,
`action.smoothness_1`, and `action.smoothness_2` results. Episode tables retain
metric version, configuration hash, unit, status, reason, model method,
quantization and checkpoint identity. Overview summaries aggregate the
available episode values by run and task; grouped summaries add run, model and
whole-group scopes. Boxplots compare episode distributions without combining
the three metrics onto a misleading shared scale. `unavailable`,
`insufficient_data`, and `error` counts remain separate, and no missing value is
converted to zero.

`descriptive-statistics.csv` contains `n`, non-finite count, minimum, P05,
median, mean, sample standard deviation, P95, and maximum. Episode files compute
these values for one episode; isolated overview rows aggregate the complete run;
grouped tables include both one row set per run and a complete-group aggregate.

Useful future extensions for larger studies are paired-seed difference plots,
bootstrap confidence intervals over tasks, normalized area-under-success-curve,
per-task critical-difference diagrams, and spatial occupancy or trajectory
density plots. These require enough matched tasks/seeds or workspace evidence
and therefore are not synthesized from insufficient runs.

### Metadata and verification

Every export has one `metadata.json` containing:

- source run and episode IDs, or group name and run names;
- experiment identity;
- model and checkpoint identity;
- source configuration hashes and normalized group selection;
- template and export-engine version;
- source/export datetimes;
- omitted conditional figures;
- file checksums used for verification.

Verify either branch with:

```bash
./ovlab export verify --kind isolated --name RUN_ID --json
./ovlab export verify --kind grouped --name GROUP_NAME --json
```

Every selected run must pass canonical verification. Empty selections, mixed
benchmark suites, or incompatible metric versions, configuration hashes, and
units fail rather than being silently combined. The older `export generate
--spec ...` command remains only as a compatibility bridge to grouped export.

## Production container workflow

The source-tree launcher runs every `report` and `export` command in the
dedicated `ovlab-reporting` image by default. It does not require an activated
Conda environment and never reuses the LIBERO benchmark image. Set
`OVLAB_REPORTING_RUNTIME=host` only for an explicitly prepared diagnostic Python.

Build or refresh only this image with:

```bash
bash deploy/scripts/build-images.sh reporting
```

Prepare an untracked deployment environment once:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Set at least the real `OVLAB_RUNS_ROOT`, `OVLAB_DERIVED_ROOT`,
`OVLAB_EXPORTS_ROOT`, and host artifact GID. Then publish a report and isolated
export:

```bash
OVLAB_REPORT_RUN_ID="$RUN_ID" \
docker compose \
  --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml \
  --profile reporting \
  run --rm reporting
```

Run explicit verification inside the production image by overriding the service
command:

```bash
docker compose \
  --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml \
  --profile reporting \
  run --rm reporting \
  report verify --run "$RUN_ID" --profile libero-task-default --build DERIVED_BUILD_ID
```

Generate and verify a grouped export:

```bash
docker compose \
  --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml \
  --profile export \
  run --rm export \
  export grouped --name paper-ablation --runs RUN_ID_A RUN_ID_B

docker compose \
  --env-file deploy/compose/.env \
  -f deploy/compose/compose.yaml \
  --profile export \
  run --rm export \
  export verify --kind grouped --name paper-ablation
```

The reporting service mounts `configs:ro`, `runs:ro`, `derived:rw`, and
`exports:rw`. The export service uses the same reporting image and mounts
`configs:ro`, `runs:ro`, and `exports:rw`; its legacy
Compose default additionally mounts an export specification read-only. Both run
as non-root, use a read-only root filesystem, drop all
capabilities, and have no network, GPU, policy socket, or checkpoint mount.

## Troubleshooting

### Source run is unavailable

Confirm that the run ID is a direct child of `OVLAB_RUNS_ROOT` and that the
directory contains its canonical manifests and `integrity.json`:

```bash
./ovlab run verify "$OVLAB_RUNS_ROOT/$RUN_ID"
```

The machine-readable error code is `source_unavailable` and the process exits
with code `4`.

### Configuration or profile is rejected

Unknown keys, invalid schemas, unsafe paths, or unsupported templates exit with
code `3`. Validate that the report profile uses `ovlab.report-profile/v1`, the
export kind is `isolated` or `grouped`, and every name is portable.

### Renderer dependency is unavailable

Local report generation requires the locked Jinja2 renderer and PDF generation
requires the locked Matplotlib backend. Use the production reporting container
when the host Python environment does not contain them. Renderer failures exit
with code `6` and do not affect the benchmark result.

### Integrity verification fails

An integrity failure exits with code `7`. Determine whether the canonical source
or a derived/export file changed. Never edit raw traces, manifests, original
metric results, or `integrity.json` to make verification pass.

### Report exists but a video link is broken

Keep the canonical `runs/` and regenerated `derived/` directories as siblings
under one data root. Verify that the episode contains both `video.mp4` and its
canonical `video.json`, then run `ovlab report verify` for the affected build.

## Operator checklist

Before sharing a result:

1. run `ovlab run verify` on every canonical source run;
2. regenerate the report from its profile or the export from its template and selection;
3. retain the derived report build ID where applicable;
4. run `ovlab report verify` with that build ID or `ovlab export verify` with the export kind and name;
5. confirm that source run IDs and checksums appear in the report manifest or export metadata;
6. share `exports/` for publication artifacts and preserve `runs/` separately as
   the immutable evidence needed for reproduction.

For the complete CLI reference, see
[`code/apps/benchctl/CLI_README.md`](code/apps/benchctl/CLI_README.md). Deployment
security and mount details are documented in [`deploy/README.md`](deploy/README.md).
