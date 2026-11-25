# Independent CPU benchmark and policy example

This project supplies a benchmark through the public `BenchmarkPlugin` API. The
separate [policy project](../policy-plugin) supplies an API v2 constant policy.
Both run on CPU with a two-joint action space. The three tasks use opaque IDs,
nonconsecutive indices and distinct terminal outcomes. No model weights,
simulator, renderer, demonstration dataset or GPU is needed. The fixture checks
integration; its task scores have no scientific interpretation.

The benchmark owns its native action interface, task catalog, runtime adapter,
configs and Dockerfile. Its control entrypoint uses only the standard library
and benchmark SDK; NumPy and adapter contracts load when settings/capabilities
are resolved or the worker constructs an adapter. Neither project imports the
other project, test helpers or OVLAB orchestration modules.

Run these commands from the OVLAB repository root. Docker with BuildKit and the
[OVLAB host environment](../../ENVIRONMENT.md) are required for container smoke.
The example images reuse `ovlab-reporting:local` as a development base. Their
recipes install the current core packages and record their own plugin identity.
A production integration should own a digest-pinned base and dependency locks.

```bash
python3 examples/benchmark-plugin/configure.py --dry-run
python3 examples/benchmark-plugin/configure.py
export OVLAB_EXTENSION_REGISTRY="$PWD/configs/extensions/cpu-example.json"
export OVLAB_LOCAL_PROFILE="$PWD/configs/local/cpu-plugin-example.yaml"
export OVLAB_COMPOSE_ENV_FILE="$PWD/configs/local/cpu-plugin-example.env"
unset OVLAB_EXECUTION_PROFILE

./ovlab plugin list
./ovlab plugin check example-policy
./ovlab plugin check example-benchmark
bash deploy/scripts/build-images.sh reporting
./ovlab plugin build --config configs/evaluation/cpu-plugin-example.yaml --dry-run
./ovlab plugin build --config configs/evaluation/cpu-plugin-example.yaml
./ovlab eval run configs/evaluation/cpu-plugin-example.yaml --offline --dry-run
./ovlab eval run configs/evaluation/cpu-plugin-example.yaml --offline
```

`configure.py` creates four namespaced files and refuses to overwrite edits.
It copies the example-only neutral registry, creates a thin experiment override,
and records absolute local checkpoint/data paths. It leaves the root registry
unchanged. The default data destination is `/tmp/ovlab-cpu-plugin-demo`; pass
`--data-root /absolute/path` before setup to choose another destination.
The local profile selects CPU and the bundled checkpoint explicitly; the Compose
environment places runs/exports under the same data root. This benchmark has no
renderer execution profile, so clear any inherited `OVLAB_EXECUTION_PROFILE`.

The public evaluation command starts the policy and benchmark as separate
containers, runs three short episodes and invokes the reporting container.
Its JSON output identifies the canonical run and published report. Verify that
run and independently recompute its metrics:

```bash
./ovlab eval verify /tmp/ovlab-cpu-plugin-demo/runs/evaluate/RUN_DIRECTORY
./ovlab report recompute /tmp/ovlab-cpu-plugin-demo/runs/evaluate/RUN_DIRECTORY
./ovlab report verify /tmp/ovlab-cpu-plugin-demo/runs/evaluate/RUN_DIRECTORY
```

Replace `RUN_DIRECTORY` with the directory printed by evaluation; the data-root
override changes this prefix. The current built-in `libero-task-default` report
profile is reused unchanged: this phase changes metadata ingress and does not
introduce a report-provider SDK.

`plugin build --config` selects both image roles for an evaluation. A standalone
policy deployment selects only the policy role. No core registry edit is needed
when either project is replaced with another compatible plugin. The generated
Compose services must have no GPU reservation, EGL/display environment, X11
mount or dataset mount.

The canonical task IDs are `calibration:alpha`, `reach:omega` and
`stability@hold`. IDs are opaque strings; suite and benchmark identity come from
catalog/manifest metadata. The second task intentionally reaches the episode
limit, while the other tasks succeed. Their terminal observations and signals
remain part of the recorded trace.

After the smoke, Compose removes the evaluation services automatically. Keep or
remove the generated `configs/extensions/cpu-example.json`,
`configs/evaluation/cpu-plugin-example.yaml`, `configs/local/cpu-plugin-example.*`
and the selected data directory as desired. Clear the three exported OVLAB
variables before returning to another registry/profile.
