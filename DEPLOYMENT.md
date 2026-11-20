# Model deployment

`ovlab deploy` builds and runs a persistent policy service. Deployment does not
start LIBERO, collect evaluation traces, or generate evaluation reports. Use
`ovlab eval` for those operations and `ovlab train` for training.

All three domains accept a YAML name or explicit path as `run TARGET` and support
`inspect TARGET`, `verify TARGET`, and `run TARGET --dry-run`:

| Domain | Default configuration directory | Inspect/verify target |
| --- | --- | --- |
| `train` | `plugin://MODEL/train/default.yaml` or a user profile | Training run path, ID, or hash |
| `eval` | `examples/experiments/` or a user scenario | Evaluation run path, ID, or hash |
| `deploy` | `plugin://MODEL/deploy/default.yaml` or `configs/deploy/` | Deployment ID or deployment YAML name/path |

Policy types, source commits and image recipes are supplied by the
[plugin registry](docs/POLICY_PLUGINS.md). A new registered policy uses the same
commands below; its plugin owns model validation and adapter construction.

## Starting and managing a service

```bash
./ovlab deploy run plugin://openvla/deploy/default.yaml --dry-run
./ovlab deploy run plugin://openvla/deploy/default.yaml
./ovlab deploy inspect openvla-libero10
./ovlab deploy verify openvla
./ovlab deploy stop openvla --dry-run
./ovlab deploy stop openvla
```

`run` builds only the image required by the policy, resolves and verifies its
checkpoint, starts a detached Docker container, and waits for model readiness.
The service stays available after the command exits and between client sessions.
An existing container is preserved; stop it before deploying the same ID again.
`stop` removes the service container and retains its deployment record.

`inspect` reports the container, model configuration, state, and published ports.
`verify` checks container ownership and identity, image identity, port bindings,
configuration bundle hashes, and a model-readiness RPC. It does not request a
prediction. Deployment records and the mounted configuration bundle live below
`<OVLAB_MODEL_DATA_ROOT>/deployments/<id>/`, with the same data-root fallback as
training. `run --dry-run` performs no build, download, Docker call, or directory write.

## Deployment YAML

Example: [`plugins/openvla/configs/deploy/default.yaml`](plugins/openvla/configs/deploy/default.yaml).

```yaml
schema_version: ovlab.deployment/v1
kind: deployment
id: my-openvla
components:
  policy: plugin://openvla/policy/lora.yaml
resources:
  registry: plugin://openvla/registry/default.yaml
build: true
runtime:
  quantization: 4bit
  gpu: all
service:
  host: 127.0.0.1
  port: 8000
  container_port: 8000
  restart: unless-stopped
  readiness_timeout_s: 900
input:
  height: 224
  width: 224
```

Policy and registry references explicitly select `plugin://`, `ovlab://`, or a user configuration under `configs/`. They determine the
model, processor, action contract, cameras, and normalization. No benchmark or
protocol config is required. The policy owner validates its model-specific input
contract; `input.height` and `input.width` default to 224.

Optional settings:

- `build` defaults to `true`. Set `false` to use an already-built compatible image.
- `image` overrides the role's default image tag. With `build: true`, the built
  role image is tagged with this name before the container starts.
- `runtime.quantization` defaults to the policy setting and accepts `none`,
  `8bit`, or `4bit`. QuIC requires `none`. `runtime.gpu` is passed to Docker's
  `--gpus` option and defaults to `all` (for example, `device=0`).
- `service.host` defaults to `127.0.0.1`; `port` and `container_port` default to
  8000. Set `port: null` to omit host publication. `host` accepts an IPv4 address.
- `service.restart` defaults to `unless-stopped`; `no`, `always`, and `on-failure`
  are also accepted. `readiness_timeout_s` defaults to 900.

CLI overrides affect one deployment:

```bash
./ovlab deploy run plugin://openvla/deploy/default.yaml --quantization 8bit --port 8123
./ovlab deploy run plugin://openvla/deploy/default.yaml --offline
./ovlab deploy run plugin://openvla/deploy/default.yaml --profile configs/local/profile.yaml
```

`--profile` selects optional machine-local checkpoint overrides; otherwise
`OVLAB_LOCAL_PROFILE` is used. Checkpoints are mounted read-only and model
loading uses offline Hugging Face settings. The configuration bundle contains
the effective quantization and its identity, while source YAML files remain
unchanged. QuIC deployment additionally uses `OVLAB_QUIC_ADAPTER_HOST_PATH` or
the adapter directory declared in the policy.

## Connecting to the model

Published ports carry the existing framed **OVLAB policy RPC protocol**, not HTTP.
The same initialize/reset/predict/end operations are available through
`TcpPolicyClient`:

```python
from ovlab_remote_policy import TcpPolicyClient, RemotePolicyAdapter

client = TcpPolicyClient("127.0.0.1", 8000)
policy = RemotePolicyAdapter(client)
# Use the ordinary PolicyAdapter lifecycle with RunContext,
# EpisodeContext, and PolicyObservation instances.
```

Closing a client session releases its episode and keeps the deployed model loaded.
The internal service commands also support TCP diagnostics:

```bash
ovlab service health --host 127.0.0.1 --port 8000 --json
```

Evaluation continues to use private AF_UNIX connections between its policy and
benchmark containers. See [`RUN.md`](RUN.md) for evaluation and
[`TRAINING.md`](TRAINING.md) for training.
