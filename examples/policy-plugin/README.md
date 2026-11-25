# Develop an OVLAB policy plugin

This separate project connects a constant-action CPU model to OVLAB through
`PolicyPluginV2`. It demonstrates neutral extension registration, static
capability description, checkpoint identity, adapter lifecycle, image builds and
persistent RPC serving. Predictions have no scientific evaluation value. The
bundled checkpoint is a tiny metadata file; no model weights are downloaded.

`extension.json` declares provider `plugin:policy:example-policy/constant`.
`src/example_policy/plugin.py` validates and resolves model settings using the
canonical `BenchmarkContext`; it never reads another plugin's YAML. The model
owns its two-joint delta action contract. `describe_capabilities()` describes
the same contract without constructing an adapter; runtime initialization uses
that description. Input requirements are empty because this constant policy
does not inspect images or state. Standalone serving has no benchmark peer.

For the complete public Compose evaluation and report pipeline, use the
[independent CPU benchmark example](../benchmark-plugin). It supplies three
opaque task IDs and a compatible native action interface. Neither project
imports the other project or test helpers.

For standalone serving, run from the OVLAB repository root. Docker with BuildKit and the
[host Python environment](../../ENVIRONMENT.md) are required. Production
projects should own digest-pinned bases and dependency locks.

Build the shared development base first, or reuse a compatible
`ovlab-reporting:local` image. The paired example registry declares this recipe;
the policy-only registry intentionally contains just the policy project:

```bash
OVLAB_EXTENSION_REGISTRY="$PWD/examples/benchmark-plugin/registry.json" \
  bash deploy/scripts/build-images.sh reporting
```

Generate the four user overrides, then select the policy-only registry:

```bash
python3 examples/policy-plugin/configure.py --dry-run
python3 examples/policy-plugin/configure.py
export OVLAB_EXTENSION_REGISTRY="$PWD/examples/policy-plugin/registry.json"
./ovlab plugin list
./ovlab plugin check example-policy
./ovlab plugin build example-policy --dry-run
./ovlab plugin build example-policy
export OVLAB_MODEL_DATA_ROOT="${TMPDIR:-/tmp}/ovlab-policy-plugin-demo"
./ovlab deploy run example-plugin --offline \
  --profile configs/local/example-plugin.yaml --dry-run
./ovlab deploy run example-plugin --offline \
  --profile configs/local/example-plugin.yaml
./ovlab deploy verify example-plugin
./ovlab deploy stop example-plugin
```

The generator creates thin policy/deployment/resource overrides and an absolute
local checkpoint override. It is idempotent and refuses to overwrite edits.
`--profile configs/local/example-plugin.yaml` explicitly selects that checkpoint
override and CPU device; standalone serving has no benchmark execution profile.
The example-only `ovlab.extensions/v1` registry leaves root registration
unchanged. Port 8123 carries OVLAB's framed policy RPC protocol. Use
`TcpPolicyClient` and `RemotePolicyAdapter` to reset episodes and request
predictions; see [DEPLOYMENT.md](../../DEPLOYMENT.md#connecting-to-the-model).

The Docker recipe copies the registered project from
`external/extensions/example-policy/self` in the isolated build context. Build
provenance records the exact selected project, source closure, config bundle and
image recipe. Rebuild after changing implementation files. `stop` removes the
service container and preserves the deployment record under the selected data
root.

To adapt this project, choose unique project/provider/config/image identifiers,
replace the native action and input contracts, then implement model loading,
reset, inference and cleanup in `runtime.py`. Keep runtime dependencies and
weight loading out of the control entrypoint. Register real checkpoint hashes
separately from implementation sources. Immutable normalization and codec
restrictions belong to the policy integration. See the
[plugin developer guide](../../docs/POLICY_PLUGINS.md) for the wider contracts.

Clear `OVLAB_EXTENSION_REGISTRY` and `OVLAB_MODEL_DATA_ROOT` after the example to
return to your usual registry and data root.
