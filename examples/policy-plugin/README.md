# Develop an OVLAB policy plugin

This is a separate Python project that connects a constant-action model to
OVLAB. It demonstrates registration, settings, adapter lifecycle, verified
artifacts, image builds and persistent RPC serving. Its predictions have no
scientific evaluation value. The bundled checkpoint is a tiny metadata file;
this example never downloads model weights.

You need an OVLAB checkout, its [host Python environment](../../ENVIRONMENT.md),
and Docker with BuildKit. Run all commands below from the OVLAB repository root.
No GPU or LIBERO dataset is needed for this service example. The development
Dockerfile reuses the reporting image's Python environment, then installs this
checkout's current OVLAB packages and this plugin. A production plugin should
own its digest-pinned base image and locked dependencies.

## Files to start with

| File | Purpose |
| --- | --- |
| `registry.json` | Binds `example_policy` to this local project and the `policy-example` image recipe. |
| `src/example_policy/plugin.py` | Implements API v1 validation, settings, adapter construction and identity. |
| `src/example_policy/runtime.py` | Implements the stateful `PolicyAdapter` lifecycle and predictions. |
| `configs/policy.yaml` | Selects the plugin, primary checkpoint and shared action contract. |
| `configs/resources.yaml` | Pins the bundled artifact's revision and file hashes. |
| `configs/deployment.yaml` | Configures a CPU service on localhost port 8123. |
| `configure.py` | Copies namespaced configs into OVLAB and writes the absolute local checkpoint override. |
| `Dockerfile` | Installs the integration and records its own build provenance. |

## Run the complete example

Prepare the small configurations. The setup is idempotent and refuses to
overwrite files you edited. `--dry-run` previews all four destination paths.

```bash
python3 examples/policy-plugin/configure.py --dry-run
python3 examples/policy-plugin/configure.py
```

Build the shared development base using OVLAB's default registry. Skip this step
if you already have a compatible `ovlab-reporting:local` image.

```bash
OVLAB_PLUGIN_REGISTRY=code/apps/benchctl/src/ovlab_benchctl/plugins/registry.json \
  bash deploy/scripts/build-images.sh reporting
```

Select this example's registry, inspect the registration, and build its image.
The variable is also required for the later `deploy` commands; `--registry` on
one `plugin` command affects only that command. This registry is a complete
replacement, so it lists only the example plugin.

```bash
export OVLAB_PLUGIN_REGISTRY="$PWD/examples/policy-plugin/registry.json"
./ovlab plugin list
./ovlab plugin inspect example-policy
./ovlab plugin sync example-policy
./ovlab plugin build example-policy --dry-run
./ovlab plugin build example-policy
```

`sync` prepares a reusable development snapshot and is optional before a build.
Builds use a temporary context by default, without reading or changing the
checkout's `external/`. The same build command works after replacing the local
source with a Git URL and commit; that form fetches the pinned revision. Use
`--workspace-sources` explicitly to build from a synced development workspace.

Start and verify the service. `build: false` in the deployment configuration uses
the image you just built; OVLAB checks its role and registered source identity.
`--offline` uses the bundled artifact through the generated profile. Runtime
mounts contain only the selected configuration bundle and verified checkpoint.

```bash
export OVLAB_MODEL_DATA_ROOT="${TMPDIR:-/tmp}/ovlab-plugin-demo"
./ovlab deploy run example-plugin --offline \
  --profile configs/local/example-plugin.yaml --dry-run
./ovlab deploy run example-plugin --offline \
  --profile configs/local/example-plugin.yaml
./ovlab service health --host 127.0.0.1 --port 8123 --json
./ovlab deploy verify example-plugin
./ovlab deploy stop example-plugin
```

Port 8123 carries OVLAB's framed policy RPC protocol. Use `TcpPolicyClient` and
`RemotePolicyAdapter` to reset episodes and request predictions; it is not an
HTTP endpoint. See [DEPLOYMENT.md](../../DEPLOYMENT.md#connecting-to-the-model)
for the client connection. `stop` removes the service container and keeps the
record below `$OVLAB_MODEL_DATA_ROOT/deployments/example-plugin/`.

The generated configs are `configs/policies/example-plugin.yaml`,
`configs/resources/example-plugin.yaml`, `configs/deploy/example-plugin.yaml`
and `configs/local/example-plugin.yaml`. They are developer examples you can
edit. The source snapshot in `external/example-policy` and the built image are
intentional caches. Clear `OVLAB_PLUGIN_REGISTRY` before returning to the default
registry, and clear `OVLAB_MODEL_DATA_ROOT` to restore your normal data root.

## Adapt it to a real model

1. Copy the project and choose a unique package, policy type, plugin ID, image
   role and Compose profile. Keep those names consistent in the registry/configs.
2. Implement your settings validation in `plugin.py`. Control hooks execute on
   the host; keep model dependencies and weight loading inside the adapter's
   `_initialize` method.
3. Replace `runtime.py` with your model's initialization, reset, inference and
   cleanup logic. Declare truthful input requirements, action specification and
   supported horizons. Return checkpoint/method identity through RPC.
4. Register real checkpoint hashes separately from model implementation sources.
   Install runtime dependencies in your own Docker recipe.
5. For distribution, change `sources.example-policy` from `kind: local` plus
   `path` to `kind: git` plus `url` and a full immutable `commit`. Source IDs
   determine `external/<id>`; update the Dockerfile's `COPY` paths too.
   Your adapter and upstream model may also be two separate registered projects:
   list both source IDs in the plugin and build recipe, and keep `entrypoint`
   pointing to your adapter. Either project can use a Git pin or a local path.

After changing local implementation files, rebuild the plugin image before
redeploying. After importing a plugin in a long-lived Python process, restart
that process before selecting a different checkout of the same Python package.

The full [plugin developer guide](../../docs/POLICY_PLUGINS.md) describes the
API, build/image identity, observations, checkpoints, mounts and migration. The
inference API covers build, serving and evaluation; model-specific training
algorithms have their own specialized components.
