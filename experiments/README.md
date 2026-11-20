# Local experiments

Keep personal experiments and study recipes here. Everything except this README
is ignored by Git. The small public verification set lives in
[`examples/experiments/`](../examples/README.md).

Start by copying one of those examples, or a plugin's `examples/train.yaml` or
`examples/evaluate.yaml`, into this directory. Use explicit references to model
defaults (`plugin://...`), OVLAB defaults (`ovlab://...`) and any local overrides
under `configs/`.

```bash
ovlab run experiments/my-experiment.yaml --plan
ovlab run experiments/my-experiment.yaml
```

`--plan` resolves composition and checks compatibility without starting GPU
workloads. A workflow can use plugin defaults directly as `config.train.profile`
or `config.evaluate.profile`; its patches supply dataset, benchmark, reporting
and resource choices. `checkpoint_from` explicitly connects a training phase's
finalized output to evaluation. Reporting is configured by the experiment.

Workflow documents use `schema_version: ovlab.experiment/v1` and `kind: workflow`.
That schema describes orchestration, independently of a plugin manifest's schema
and the schemas of its individual defaults. See the
[workflow guide](../docs/EXPERIMENT_WORKFLOW.md) for phase inheritance and evidence.
