# OVLAB configuration

This directory contains versioned defaults owned by OVLAB. Use an explicit
`ovlab://` reference, for example `ovlab://metrics/default.yaml`. Model-specific
defaults belong to registered plugin projects and use `plugin://` references.

| Directory | Owner and purpose |
| --- | --- |
| `artifacts/` | Artifact storage settings. |
| `benchmarks/` | Current benchmark integrations; candidates for future plugins. |
| `exports/` | Report export selection and formats. |
| `interfaces/` | Shared observation/action contracts. |
| `local/` | A machine-profile example to copy into the local workspace. |
| `metrics/` | Report metrics and their required signals. |
| `policies/` | A model-free mock policy for descriptor validation and the CPU test scaffold. |
| `profiles/` | Renderer execution settings. |
| `protocols/` | Evaluation limits, recording and reproducibility settings. |
| `registry/` | Empty resources for model-free runs. |

The model plugin selects its policy, checkpoint catalog and deployment defaults.
An experiment selects the benchmark, protocol, metrics and artifact store. Metric
and benchmark configuration remains here until those integrations become
plugins with the same explicit composition mechanism.

Create local extensions in [`configs/`](../configs/README.md), rather than editing
these defaults. Files in `configs/` and `experiments/` are ignored by Git. Only
the documented examples under [`examples/`](../examples/README.md) belong to the
public verification set.

The small `protocols/smoke.yaml` protocol verifies execution and reporting. Its
two-step episode limit cannot measure task performance. Real experiments should
select or extend `protocols/default.yaml` and set their own limits explicitly.
