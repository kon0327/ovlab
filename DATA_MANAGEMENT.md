# OVLAB data management

The `ovlab data` command manages canonical benchmark runs, regenerable derived
reports, and publication exports from the host. It does not start Docker and
does not require a Conda environment.

The default roots are:

```text
OVLAB_DATA_ROOT/
├── runs/       # canonical benchmark runs
├── derived/    # benchmark and training reports
├── exports/    # isolated and grouped publication exports
└── archive/
    ├── runs/
    ├── derived/
    ├── exports/
    └── manifests/
```

`OVLAB_DATA_ROOT` defaults to the repository sibling `ovlab-data/`. Existing
`OVLAB_RUNS_ROOT`, `OVLAB_DERIVED_ROOT`, and `OVLAB_EXPORTS_ROOT` overrides
remain authoritative. `OVLAB_ARCHIVE_ROOT` may override only the archive
destination.

## List data

List active runs, reports, and exports:

```bash
./ovlab data list
./ovlab data list --kind runs
./ovlab data list --kind reports
./ovlab data list --kind exports
```

Run rows include the short lookup hash displayed by the CLI:

```text
run <run-id> (<run-hash>) <state> <path>
<other-kind> <id> <state> <path>
```

The hash is the eight-character suffix of current readable run IDs. Legacy run
IDs receive a stable compatibility alias derived from their ID. It is a lookup
alias, not the scientific configuration hash or artifact-integrity checksum.
Every benchmark-run selector accepts either the full run ID or this displayed
hash. Exact IDs take precedence; a missing or colliding hash fails clearly.

Use `--detail` for file counts and byte sizes, or `--json` for the stable CLI
envelope. List archived entries with:

```bash
./ovlab data list --archived
```

Benchmark report IDs equal their canonical run IDs. Training report IDs use an
explicit namespace:

```text
training:<training-run-id>
```

Export IDs are namespaced by their on-disk branch to avoid ambiguity:

```text
isolated:<run-id>
grouped:<group-id>
```

## Preview every mutation

Both archive and delete support a non-mutating preview:

```bash
./ovlab data archive --run RUN_ID --dry-run
./ovlab data archive --run RUN_HASH --dry-run
./ovlab data delete --report RUN_ID --dry-run
./ovlab data archive --export isolated:RUN_ID --dry-run
./ovlab data archive --all --dry-run --json
```

The preview resolves and validates the complete target set before changing any
file.

## Archive

Archive one run, report, or export:

```bash
./ovlab data archive --run RUN_ID
./ovlab data archive --run RUN_HASH
./ovlab data archive --report RUN_ID
./ovlab data archive --report training:TRAINING_RUN_ID
./ovlab data archive --export isolated:RUN_ID
./ovlab data archive --export grouped:GROUP_ID
```

Archive all benchmark runs, derived reports, and exports:

```bash
./ovlab data archive --all
```

The complete selection is refused if any run, report, or export is active or
incomplete; no target is silently skipped.

Interactive execution asks for an explicit confirmation. Automation must use
`--yes`; JSON-mode mutations never prompt:

```bash
./ovlab data archive --all --yes --json
```

Archive paths mirror the active layout:

```text
archive/
├── runs/<run-id>/
├── derived/<run-id>/
├── derived/training/<training-run-id>/
├── exports/isolated/<run-id>/
├── exports/grouped/<group-id>/
└── manifests/
    ├── run/<run-id>.json
    ├── report/<report-id>.json
    └── export/<export-id>.json
```

Because `runs/` and `derived/` remain siblings under `archive/`, relative links
from an archived benchmark report to an archived canonical video remain valid
when both are selected by `--all`. An existing archive destination is never
overwritten. Same-filesystem moves are atomic; a cross-filesystem fallback
copies into a staging directory, verifies the complete content inventory, then
publishes and removes the source.

## Delete

Delete one finalized run, completed report, or completed export:

```bash
./ovlab data delete --run RUN_ID
./ovlab data delete --report RUN_ID
./ovlab data delete --export isolated:RUN_ID
./ovlab data delete --export grouped:GROUP_ID
```

Delete every benchmark run, derived report, and export:

```bash
./ovlab data delete --all
```

As with archival, an active or incomplete item refuses the complete operation.

Deletion is irreversible and always requires interactive confirmation or
`--yes`. Deleting a run does not implicitly delete its report or isolated
export, and deleting a report or export does not delete its canonical run. Use
`--all` only after reviewing its dry-run selection.

## Safety boundaries

The manager:

- refuses active or incompletely finalized runs;
- refuses incomplete or `.partial` report and export builds;
- rejects path traversal and symbolic links;
- never overwrites an existing archive entry;
- never follows a selected path outside its configured root;
- never modifies `datasets/`, `checkpoints/`, `training-runs/`, or the source
  repository;
- does not delete archived data. Archive retention and restore commands remain
  separate future operations.

Before deletion, OVLAB checks write and execute permission on every selected
directory. The complete operation is rejected before `rmtree` starts if even
one directory is not deletable by the current user. This prevents a multi-run
`--all` request from deleting earlier targets and failing later because of a
container-created ownership mismatch.

Finalized container artifacts use mode `2770` for directories and `0640` for
files. The host artifact group can therefore traverse and remove a finalized
tree, while raw artifact files remain non-writable to group members. Container
processes use a restrictive temporary umask during publication and seal the
final tree to these modes before reporting success.

If a benchmark or reporting process is running, let it finish before invoking
data management. The artifact-state checks are a safety guard, not a substitute
for coordinating concurrent operators.
