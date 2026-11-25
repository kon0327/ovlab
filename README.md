# OpenVLABenchmark (OVLAB)

OpenVLABenchmark (OVLAB) is a reproducible experimental framework for evaluating Vision–Language–Action policies through independent model plugins, initially OpenVLA with the LIBERO benchmark.

OVLAB is authored by **Jiří Konečný**, VSB – Technical University of Ostrava.

Current project version: **0.2.0**. See the complete
[`release notes`](RELEASE_NOTES.md) for implemented capabilities, compatibility
boundaries and current scope limits.

The complete system design, runtime boundaries, data flows and extension points
are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

OVLAB follows a **Define experiment → Plan → Run → Report** workflow: a YAML or JSON
experiment tree can contain models, tasks and ablations at any depth. Each node
can run independently and produces immutable canonical evidence plus a
hierarchical report. See the [experiment workflow guide](docs/EXPERIMENT_WORKFLOW.md)
and example [pipeline smoke](examples/experiments/plugin-pipeline-smoke.yaml).

For visual authoring, run `ovlab editor` to open the static
[Workflow Studio](code/apps/workflow-builder/index.html) in a browser. It offers
the repository's real profiles, hierarchical nodes, checkpoint handoff and YAML
export. See its [usage guide](code/apps/workflow-builder/README.md).

The experiment runner and benchmark adapter execute together in a runner process or container. Each VLA implementation runs independently as a policy service, communicating through a dependency-light policy protocol. This separation allows Vanilla, LoRA, OFT, and QuIC implementations to keep distinct dependency environments.

Model and benchmark implementations connect through independent, versioned SDKs
and one [extension registry](docs/EXTENSION_ARCHITECTURE.md). The registry binds
each project to a Git commit or local directory; its manifest owns typed provider
declarations, source pins, defaults and image recipes. Existing model projects
retain `plugin.json`; the [LIBERO project](plugins/libero/README.md) uses
`extension.json` and owns its benchmark, renderer and dataset provider. The
[policy developer guide](docs/POLICY_PLUGINS.md) describes model integration.
Register a project
with `./ovlab plugin register PATH_OR_GIT_URL` (add `--commit FULL_COMMIT` for Git),
then use `./ovlab plugin list`, `./ovlab plugin sync` and
`./ovlab plugin build PLUGIN`. Builds use a temporary context prepared from
registered Git commits or local projects; `external/` is a development workspace
and optional offline Git cache. A new policy family needs no core dispatch or
Compose changes. The [CPU policy example](examples/policy-plugin/README.md)
demonstrates persistent serving; the independent
[CPU benchmark example](examples/benchmark-plugin/README.md) combines both
providers through evaluation, canonical metrics and report/export.

## Repository layout

- `code/`: generic core packages, policy/benchmark SDKs, metrics, applications, and tests.
- `plugins/`: independent OpenVLA, OpenVLA-OFT, OpenVLA-QuIC and LIBERO projects, with reusable model integration code under `_shared/`.
- `config/`: tracked generic OVLAB interface, metric, protocol and artifact defaults, addressed with `ovlab://`; concrete benchmark defaults belong to their project.
- `configs/`: ignored user extensions of plugin or OVLAB defaults.
- `experiments/`: ignored user experiment specifications.
- `examples/experiments/`: the small maintained verification suite.
- `deploy/`: reserved for Docker, Compose, and deployment scripts.
- `external/`: ignored source workspace populated from registered Git commits or local projects.
- `references/`: tracked historical source artifacts and their provenance.
- `runs/<experiment-id>/train|evaluate/`: canonical evidence for experiment
  nodes; contents are not versioned. Legacy direct commands retain their old
  `runs/train|evaluate/` locations.

Container deployments use a host-backed artifact workspace outside the source
checkout. The canonical convention is `ovlab-data/runs` for immutable evidence,
`ovlab-data/checkpoints` for orchestrator-managed model artifacts,
`ovlab-data/datasets` for benchmark datasets, and `ovlab-data/exports` for
regenerated reports, dashboards, and curated publication outputs. Existing snapshots in the user's global Hugging
Face cache are reused through verified hard links rather than copied. Only
benchmark containers write canonical runs; reporting mounts them read-only.
See [`deploy/README.md`](deploy/README.md).

Dataset discovery, explicit acquisition, local import, verification and storage
are documented in the [`Dataset operations guide`](DATASETS.md). Versioned
training profiles, isolated OpenVLA-LoRA training and immutable checkpoint
handoff are documented in the [`Training operations guide`](TRAINING.md).
Evaluation invocation and the difference between Docker deployment and a
machine-local run are described in [`RUN.md`](RUN.md).
Safe listing, archival and deletion of benchmark runs, derived reports, and
publication exports are documented in the [`Data management guide`](DATA_MANAGEMENT.md).

## Command line

To make the checkout's launcher available as `ovlab` without activating an
environment, install it once into `~/.local/bin`:

```bash
bash ./install-ovlab-cli.sh
ovlab --help
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
```

The installer prints a PATH instruction if `~/.local/bin` is not already on
PATH. It never replaces an unrelated command. The launcher uses the same
Docker routing and Python selection as `./ovlab`; repository-relative
experiment specifications also work from another directory.

The principal target can be positional across reporting, exports and training:

```bash
ovlab report RUN_ID
ovlab report EXPERIMENT_ID
ovlab report verify RUN_ID
ovlab report TRAINING_RUN_ID
ovlab dataset inspect DATASET_ID
ovlab checkpoint verify CHECKPOINT_ID
```

Reporting and data management retain their named selectors. `ovlab run` uses
one positional experiment target. For data archive and delete, use a typed target such as
`run:ID`, `report:ID`, or `export:isolated:ID` to avoid ambiguity.

For a dedicated host Python environment, use the environment installer. It
prefers Conda and uses `venv` when Conda is unavailable; see
[Environment setup](ENVIRONMENT.md).

```bash
source ./install-ovlab-env.sh
ovlab --help
```

The installer activates the environment in the current Bash session. Conda
users can later run `conda activate ovlab-env`. Use the unified CLI there, or
call `./ovlab` from a checkout. The launcher selects the installed host
environment when available:

```bash
./ovlab --help
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml --plan
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml
./ovlab run examples/experiments/plugin-pipeline-smoke.yaml --verify
./ovlab run examples/experiments/openvla-eval.yaml
./ovlab run EXPERIMENT_ID_OR_HASH --verify
./ovlab doctor
./ovlab data list runs
./ovlab report EXPERIMENT_ID_OR_HASH
./ovlab checkpoint list
./ovlab deploy run plugin://openvla/deploy/default.yaml --dry-run
```

`ovlab run EXPERIMENT` is the single execution entry point. The resolver reads
its schema and kind: a training profile, an evaluation scenario, or a hierarchical
workflow. A standalone profile becomes a one-node experiment. Every execution
has the same experiment metadata and canonical phase hierarchy.

`--plan` resolves and displays the tree without starting workloads. `--verify`
validates a source configuration; with a stored run or experiment ID/path/hash,
it checks canonical evidence instead. Individual runs have eight-character
lookup hashes; experiment instances have ten-character suffix hashes. Ambiguous
references are rejected. `ovlab data list runs` lists individual run IDs.

Runtime settings belong in YAML: model and quantization in policy configuration,
renderer in the selected benchmark's execution profile, and `execution.max_workers` at the workflow
root. Configuration mappings inherit from parent to child; sequences and scalar
values replace inherited values. See the [experiment guide](docs/EXPERIMENT_WORKFLOW.md).

Plugin defaults are addressed as `plugin://openvla/policy/default.yaml`.
A local YAML can `extends` a plugin URI and override just the checkpoint,
quantization or execution settings; use the local file explicitly as the
experiment component. Plugin checkpoint catalogs are declared in the manifest
and discovered by `./ovlab checkpoint list --source registry`. The generic
core registry can remain empty when no model plugin is selected.

Evaluation phases launch the selected policy service and benchmark runner in separate
Compose containers, propagate results, clean up the private deployment, and
publish reports below `exports/`. Training phases use their isolated trainer and
checkpoint finalizer. Persistent model services use `ovlab deploy`; see
[model deployment](DEPLOYMENT.md).

Set `OVLAB_LOCAL_PROFILE` to a gitignored machine profile when resolving a
machine-local checkpoint path. Installing `ovlab-benchctl` exposes the same command as the
`ovlab` console entrypoint. Complete command, output, exit-code, foreground
service, cleanup, and plugin inspection behavior is documented in the
[`OVLAB CLI guide`](code/apps/benchctl/CLI_README.md).

Report generation, offline viewing, integrity verification, isolated/grouped exports,
and the production reporting-container workflow are covered in the
[`Reporting and export operations guide`](REPORTING.md).

## Testing

OVLAB uses the lightweight `ovlab-tester` Conda environment for CPU-only automated tests. Create it from `deploy/environments/ovlab-tester/environment.yml`, then run:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh
```

The loose checkpoint revision paths have CPU-only train-plan and eval-config
fixtures. They cover an omitted revision and explicit `local/main`, downloaded
`remote/main`, remote planning without downloads, and a registered pinned
commit. Run just that matrix with:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/unit/training/test_revision_scenarios.py \
  code/tests/unit/cli/test_checkpointing.py \
  code/tests/unit/cli/test_revision_deployment_scenarios.py
```

GPU and policy-specific smoke tests remain isolated in their corresponding policy environments.

The experimental QuIC integration, portable method, isolated images and
supported migration scope are described in [QuIC migration](docs/QUIC_MIGRATION.md).
Shared training variants use the [ablation workflow](docs/ABLATIONS.md).
