# OVLAB Analytical Laboratory

The OVLAB analytical laboratory is a separate workspace for offline analysis of completed runs and generation of interactive HTML reports.

The configured canonical run directory (`OVLAB_RUNS_ROOT`, conventionally
`/home/kony/dissertation/ovlab-data/runs`) is an immutable, read-only input source
for laboratory tooling. Laboratory tooling must never modify raw episode traces,
experiment manifests, or original metric outputs. Every derived analysis must
preserve references to its source run IDs, analysis configuration, and
metric-plugin versions.

## Workspace conventions

- `notebooks/` contains exploratory research notebooks.
- `src/ovlab_lab/` contains reusable loaders, analyses, visualizations, and report-generation code.
- `queries/` contains reusable DuckDB and SQL queries.
- `templates/` and `static/` contain HTML templates, CSS, JavaScript, and other report assets.
- `reports/` contains repo-local exploratory HTML reports. Reproducible deployment
  outputs belong under `OVLAB_DERIVED_ROOT`, conventionally
  `/home/kony/dissertation/ovlab-data/derived`.

Reports must preserve links to their source run IDs and analysis configurations so that results remain traceable and reproducible. Generated reports are ignored by Git and are not committed by default.

Canonical `runs/`, regenerated `derived/`, and curated `exports/` are separate
ownership domains. Only the benchmark runtime may create canonical runs; analysis
and reporting read them without write permission. Promotion from `derived/` into
`exports/` is a deliberate host-side publication step.
