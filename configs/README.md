# Local configuration workspace

Put personal overrides and machine settings here. Everything except this README
is ignored by Git. This directory does not contain OVLAB's shipped defaults.

There are three explicit configuration origins:

| Reference | Location |
| --- | --- |
| `ovlab://metrics/default.yaml` | Versioned OVLAB defaults in `config/`. |
| `plugin://openvla/policy/default.yaml` | Defaults supplied by a registered plugin. |
| `policy/my-model.yaml` | A local file under `configs/`. |

Copy a category example from `plugins/<model>/examples/` into this directory.
Each plugin provides policy, deployment, checkpoint registry and evaluation
examples; trainable plugins also provide a training example. Its `extends`
reference keeps the connection to the plugin explicit.

For example, save this as `configs/policy/my-openvla.yaml`:

```yaml
schema_version: "0.1.0"
kind: policy
extends: plugin://openvla/policy/lora.yaml
settings:
  runtime:
    quantization: 4bit
```

Select it from your experiment with
`components.policy: policy/my-openvla.yaml`. An override has no effect until an
experiment or build configuration references it. Mappings merge recursively;
lists and scalar values replace their parent values. Relative `extends` paths
stay within their configuration origin. Use `plugin://` or `ovlab://` when
crossing origins.

For machine paths and GPU selection, copy `config/local/example.yaml` to
`configs/local/machine.yaml`, replace the example absolute paths, and select it
with the normal local-profile option. Checkpoint weights and generated run
artifacts remain outside configuration directories.

Plugin registration writes `configs/plugins/registry.json` by default. This is
also local workspace state; the public registry in `plugins/registry.json` only
provides the bundled reference projects. The same arrangement allows a future
external configuration library without adding personal experiments to OVLAB.
