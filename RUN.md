# Running Evaluations

OVLAB separates the portable evaluation experiment from machine-specific run
settings. A file in `experiments/evaluation/` defines the benchmark, policy, and
metrics. A local profile provides paths to datasets, checkpoints, and outputs,
as well as device selection. These profiles serve different purposes.

## Evaluation lifecycle

`ovlab run TARGET` selects a YAML from `experiments/evaluation`. After a run,
use `ovlab data list runs --detail` and `ovlab run RUN_ID_OR_HASH --verify`. All evaluation
run examples accept explicit YAML paths as well as short names. `deploy run` now
starts a persistent model service described in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Recommended Docker Workflow

For a normal evaluation, use `ovlab run`. Pass the experiment as a positional
argument; this workflow does not require `configs/local/profile.yaml`. If you
do not have a local Compose environment file yet, copy the template once and
set the data paths:

```bash
cp deploy/compose/.env.example deploy/compose/.env
./ovlab run \
  experiments/evaluation/libero10-lora-4bit-three-per-task.yaml --plan
./ovlab run \
  experiments/evaluation/libero10-lora-4bit-three-per-task.yaml
```

`--plan` displays and validates the plan without starting Docker. An evaluation
starts the policy service and LIBERO runner in separate containers. It
gets local paths and output locations from `deploy/compose/.env` and the
container definitions.

The Docker Compose profile (`openvla`, `oft`, or `quic`) is selected from the
experiment YAML.

## Host configuration

`ovlab run` chooses training, evaluation or a workflow from the input YAML. An
evaluation declaring `deployment` uses its configured Compose topology. Host-only
worker entry points are internal; they are not separate public run commands.
Machine paths remain in the Compose environment and optional gitignored local
profile. Select that profile with:

```bash
export OVLAB_LOCAL_PROFILE=/absolute/path/to/profile.yaml
```

Runtime options belong in YAML: `deployment.renderer`, `deployment.offline`,
policy `settings.runtime.quantization`, and workflow `execution.max_workers`.

## Selecting a Configuration

`ovlab run` accepts both explicit paths and short YAML names:

```bash
./ovlab run experiments/evaluation/libero10-lora-4bit-three-per-task.yaml
./ovlab run libero10-lora-4bit-three-per-task
```

After a run, list and inspect the results with:

```bash
./ovlab data list --kind runs
./ovlab run RUN_ID_OR_HASH --verify
```

For details about containers and their configuration, see
[`deploy/README.md`](deploy/README.md) and the
[`OVLAB CLI guide`](code/apps/benchctl/CLI_README.md).

## Weights & Biases tracking

Set `OVLAB_WANDB_PROJECT` (or `WANDB_PROJECT`) to mirror completed evaluation phases to W&B. The W&B SDK installs automatically as a dependency of
`ovlab-benchctl` through the [host CLI environment](ENVIRONMENT.md);
authenticate with `wandb login` or `WANDB_API_KEY`.
`OVLAB_WANDB_ENTITY` is optional, and standard
W&B settings such as `WANDB_MODE=offline` apply.

```bash
OVLAB_WANDB_PROJECT=ovlab ./ovlab run experiments/evaluation/libero-mock-smoke.yaml
```

The mirror records numeric episode metrics, per-task metrics, averages across
tasks, the resolved plan and final manifest. It runs on the host after the
isolated evaluation completes. The `wandb` field in the
CLI result contains the sync status and URL; W&B errors leave the canonical
evaluation result unchanged.
