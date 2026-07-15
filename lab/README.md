# OVLAB Analytical Laboratory

The OVLAB analytical laboratory is a separate workspace for offline analysis of completed runs and generation of interactive HTML reports.

The root-level `runs/` directory is an immutable, read-only input source. Laboratory tooling must never modify raw episode traces, experiment manifests, or original metric outputs. Every derived analysis must preserve references to its source run IDs, analysis configuration, and metric-plugin versions.

## Workspace conventions

- `notebooks/` contains exploratory research notebooks.
- `src/ovlab_lab/` contains reusable loaders, analyses, visualizations, and report-generation code.
- `queries/` contains reusable DuckDB and SQL queries.
- `templates/` and `static/` contain HTML templates, CSS, JavaScript, and other report assets.
- `reports/` contains generated interactive HTML reports.

Reports must preserve links to their source run IDs and analysis configurations so that results remain traceable and reproducible. Generated reports are ignored by Git and are not committed by default.
