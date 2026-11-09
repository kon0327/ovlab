# OVLAB data management

The `ovlab data` command manages canonical benchmark runs, interactive
reports, and publication exports from the host. It does not start Docker and
does not require a Conda environment.

The default roots are:

```text
OVLAB_DATA_ROOT/
├── runs/
│   └── <experiment-id>/
│       ├── train/<node-path>/<run-id>/
│       └── evaluate/<node-path>/<run-id>/
├── exports/
│   └── <experiment-id>/{train,evaluate}/<node-path>/
│       ├── isolated/   # per-run data and figures
│       ├── grouped/    # comparison exports
│       └── interactive/ # complete HTML reports
└── archive/
    ├── runs/
    ├── exports/
    └── manifests/
```

`OVLAB_DATA_ROOT` defaults to the repository sibling `ovlab-data/`. Existing
`OVLAB_RUNS_ROOT` and `OVLAB_EXPORTS_ROOT` may override the canonical roots.
Reports, dashboards and publication exports use the same `OVLAB_EXPORTS_ROOT`.
`OVLAB_ARCHIVE_ROOT` may override only the archive destination.
Older flat `runs/{train,evaluate}` and `exports/{train,evaluate}` layouts remain
readable for migration and archival operations.

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

New reports and exports use `exports/train/` or `exports/evaluate/`. Legacy flat
exports remain manageable without moving them. If the same artifact exists in
both layouts, the ordinary ID selects the typed copy; `data list` also shows a
`legacy:` ID for the older copy, such as `legacy:training:RUN` for a training
report or `legacy:isolated:RUN` for an export. Archive operations preserve each
copy’s existing relative path.

Archive paths mirror the active layout:

```text
archive/
├── runs/train/<training-run-id>/
├── runs/evaluate/<evaluation-run-id>/
├── exports/evaluate/derived/<run-id>/
├── exports/train/derived/<training-run-id>/
├── exports/evaluate/interactive/<run-id>/
├── exports/train/interactive/<training-run-id>/
├── exports/evaluate/isolated/<run-id>/
├── exports/evaluate/grouped/<group-id>/
└── manifests/
    ├── run/train/<training-run-id>.json
    ├── run/evaluate/<evaluation-run-id>.json
    ├── report/<report-id>.json
    └── export/<export-id>.json
```

Because report/dashboard files and canonical runs retain their relative layout
under `archive/`, links from an archived report to its video remain valid when
both are selected by `--all`. An existing archive destination is never
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
To deliberately delete an interrupted legacy run or another incomplete artifact,
add `--force` after first reviewing the forced dry-run:

```bash
./ovlab data delete --run RUN_ID --force --dry-run
./ovlab data delete --run RUN_ID --force
```

`--force` bypasses only the artifact-state check. It does not bypass confirmation,
root-containment checks, symbolic-link rejection, or the recursive permission
preflight, and it is not accepted by `data archive`.

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
- never modifies unrelated datasets, checkpoints, or source files;
- does not delete archived data. Archive retention and restore commands remain
  separate future operations.

A benchmark stopped by SIGINT or SIGTERM is finalized with run status `aborted`
and an `aborted` terminal trace for the in-progress episode. Such a finalized run
can be listed, inspected, archived, or deleted normally. `--force` is primarily
for older or externally killed runs that never received a final manifest.

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

# Migrating legacy runs into experiment folders

The migration utility assigns **one experiment to each existing legacy run**.
It never guesses which runs belong in a comparison. Existing grouped exports
are preserved under `exports/legacy-comparisons/{train,evaluate}/grouped`.
Old isolated exports are copied as `isolated/<run-id>--legacy`, leaving the
normal `isolated/<run-id>` name available for a fresh export.

Run the four phases from an environment with the OVLAB packages installed:

```bash
python -m ovlab_benchctl.experiment_migration preview --data-root /path/to/ovlab-data
python -m ovlab_benchctl.experiment_migration apply --data-root /path/to/ovlab-data
python -m ovlab_benchctl.experiment_migration finalize --data-root /path/to/ovlab-data
python -m ovlab_benchctl.experiment_migration cleanup --data-root /path/to/ovlab-data
```

`preview` is read-only and prints the exact source and destination mapping.
`apply` copies every run and associated export, verifies byte inventories and
completed canonical runs, writes `experiment-migration-v1.json`, and leaves
all originals in place. Each migrated run receives `.experiment` metadata and
an export `index.html` with its status. The manifest lists absolute destination
run paths, report requirements, source export paths, and the expected
interactive roots. Generate and verify the new interactive reports for every
entry with `report_required: true` before `finalize`.

`finalize` rechecks source and destination inventories and verifies required
interactive reports. It then moves only the listed legacy directories into a
migration quarantine; if a move fails, it restores the earlier moves. `cleanup`
rechecks the new artifacts and removes the quarantine. The original run and
export directories are never removed during `preview` or `apply`. If the
manifest lists unmatched legacy reports, `finalize` stops for explicit review.

Use `--manifest /path/to/manifest.json` on every command to choose a different
manifest path. Keep that path identical across `apply`, `finalize`, and
`cleanup`.
