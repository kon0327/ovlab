# Running Evaluations

OVLAB separates the portable evaluation experiment from machine-specific run
settings. A file in `configs/experiments/evaluation/` defines the benchmark, policy, and
metrics. A local profile provides paths to datasets, checkpoints, and outputs,
as well as device selection. These profiles serve different purposes.

## Evaluation lifecycle

`eval run TARGET` selects a YAML from `configs/experiments/evaluation`. After a run,
use `eval inspect RUN_ID_OR_HASH` or `eval verify RUN_ID_OR_HASH`. All evaluation
run examples accept explicit YAML paths as well as short names. `deploy run` now
starts a persistent model service described in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Recommended Docker Workflow

For a normal evaluation, use `eval run`. Pass the experiment as a positional
argument; this workflow does not require `configs/local/profile.yaml`. If you
do not have a local Compose environment file yet, copy the template once and
set the data paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
./ovlab eval run \
  configs/experiments/evaluation/libero10-lora-4bit-three-per-task.yaml --dry-run
./ovlab eval run \
  configs/experiments/evaluation/libero10-lora-4bit-three-per-task.yaml
```

`--dry-run` displays and validates the plan without starting Docker. `eval
run` starts the policy service and LIBERO runner in separate containers. It
gets local paths and output locations from `deploy/compose/.env` and the
container definitions.

The Docker Compose profile (`openvla`, `oft`, or `quic`) is selected from the
experiment YAML.

## Direct Host Execution

`ovlab run` uses the low-level native execution path. You can provide the
evaluation YAML by its short name; the CLI looks for it in
`configs/experiments/evaluation/`:

```bash
./ovlab run libero10-lora-4bit-three-per-task --dry-run
./ovlab run libero10-lora-4bit-three-per-task
```

This path requires a local profile so the resolver can access machine-specific
absolute paths and device settings. Pass it with `--profile`; otherwise the CLI
uses `OVLAB_LOCAL_PROFILE`, then `configs/local/profile.yaml`. The error
`runtime resolution requires --profile, OVLAB_LOCAL_PROFILE, or configs/local/profile.yaml`
means that the experiment YAML was found, but the machine-specific settings
are missing.

Create a local profile from the template and replace its paths with absolute
paths for your machine:

```bash
cp configs/local/profile.example.yaml configs/local/profile.yaml
```

Set `paths.checkpoint_root`, `paths.dataset_root`, `paths.runs_root`, and
`devices.primary_gpu` for your host. You can remove the example checkpoint
override under `resources.checkpoints` if you do not use it. The local profile
is gitignored and should not be committed. To store it elsewhere, set its path
with this variable:

```bash
export OVLAB_LOCAL_PROFILE=/absolute/path/to/profile.yaml
```

To select a profile for one run, use:

```bash
./ovlab run libero10-lora-4bit-three-per-task --profile configs/local/profile.yaml
```

Do not add absolute paths or GPU settings to portable experiment files. Direct
execution also requires a compatible native Python environment and runtime
dependencies; the Docker workflow is usually simpler for operator runs.

## Selecting a Configuration

`ovlab run` accepts both explicit paths and short YAML names:

```bash
./ovlab run configs/experiments/evaluation/libero10-lora-4bit-three-per-task.yaml
./ovlab run libero10-lora-4bit-three-per-task
```

After a run, list and inspect the results with:

```bash
./ovlab data list --kind runs
./ovlab eval inspect RUN_ID_OR_HASH
./ovlab eval verify RUN_ID_OR_HASH
```

For details about containers and their configuration, see
[`deploy/README.md`](deploy/README.md) and the
[`OVLAB CLI guide`](code/apps/benchctl/CLI_README.md).
