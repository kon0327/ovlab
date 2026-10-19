# Running Evaluations

OVLAB separates the portable evaluation experiment from machine-specific run
settings. A file in `configs/experiments/` defines the benchmark, policy, and
metrics. A local profile provides paths to datasets, checkpoints, and outputs,
as well as device selection. These profiles serve different purposes.

## Recommended Docker Workflow

For a normal evaluation, use `deploy run`. Pass the experiment as a positional
argument; this workflow does not require `configs/local/profile.yaml`. If you
do not have a local Compose environment file yet, copy the template once and
set the data paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
./ovlab deploy run \
  configs/experiments/libero10-lora-4bit-three-per-task.yaml --dry-run
./ovlab deploy run \
  configs/experiments/libero10-lora-4bit-three-per-task.yaml
```

`--dry-run` displays and validates the plan without starting Docker. `deploy
run` starts the policy service and LIBERO runner in separate containers. It
gets local paths and output locations from `deploy/compose/.env` and the
container definitions.

For `deploy run`, `--profile` has a different meaning: it selects the Docker
Compose profile (`openvla`, `oft`, or `quic`), not the experiment YAML. The
profile is usually selected from the experiment, so this option is not needed:

```bash
./ovlab deploy run configs/experiments/EXPERIMENT.yaml --profile openvla
```

## Direct Host Execution

`ovlab run` uses the low-level native execution path. You can provide the
evaluation YAML by its short name; the CLI looks for it in
`configs/experiments/`:

```bash
./ovlab run --profile libero10-lora-4bit-three-per-task.yaml --dry-run
./ovlab run --profile libero10-lora-4bit-three-per-task.yaml
```

This path requires a local profile so the resolver can access machine-specific
absolute paths and device settings. If `OVLAB_LOCAL_PROFILE` is unset, the CLI
looks for `configs/local/profile.yaml`. The error
`runtime resolution requires OVLAB_LOCAL_PROFILE or configs/local/profile.yaml`
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

Do not add absolute paths or GPU settings to portable experiment files. Direct
execution also requires a compatible native Python environment and runtime
dependencies; the Docker workflow is usually simpler for operator runs.

## Selecting a Configuration

`ovlab run` accepts both explicit paths and short YAML names:

```bash
./ovlab run configs/experiments/libero10-lora-4bit-three-per-task.yaml
./ovlab run --profile libero10-lora-4bit-three-per-task.yaml
```

The short-name `--profile` option applies only to the top-level `run` command.
For `deploy run`, pass the experiment positionally because its `--profile`
option is reserved for selecting a Docker Compose service.

After a run, list and inspect the results with:

```bash
./ovlab data list --kind runs
./ovlab run inspect RUN_ID_OR_HASH
./ovlab run verify RUN_ID_OR_HASH
```

For details about containers and their configuration, see
[`deploy/README.md`](deploy/README.md) and the
[`OVLAB CLI guide`](code/apps/benchctl/CLI_README.md).
