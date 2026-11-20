# Comparing QuIC and LoRA target coverage

Module coverage is a property of a research experiment, not a model-plugin
default. The old preset matrix and `train-quic-lora-coverage` helper have been
removed. Create an explicit user profile extending
`plugin://openvla-quic/train/default.yaml`, then select it from your experiment.
The [training example](../plugins/openvla-quic/examples/train.yaml) explains the
composition fields; the [pipeline smoke](../examples/experiments/plugin-pipeline-smoke.yaml)
shows checkpoint handoff and reporting.

A coverage comparison must record exact adapted paths, operation types and
parameter counts. A LoRA suffix list does not by itself define equivalent QuIC
capacity. Match full leaf names or an anchored full-path regular expression;
`proj` must not select `q_proj` merely as a substring.

The integration supports dense linear and convolutional target operations.
Convolution stride, padding, dilation, groups and bias remain properties of the
frozen host operation. LoRA's rank is distinct from a convolution's output
channel count. QuIC's compound construction acts on the selected operation's
actual dimensions.

OpenVLA can expose vision layers that do not contribute to its selected
intermediate features. Use `policy_active_only: true` when comparing active
coverage. Inspect `target-selection.json` for matches and inactive exclusions and
`target-manifest.json` for adapted paths and shapes. An unsupported active target
must fail rather than silently disappear from a coverage claim.

In a user study, explicitly select placement, target names or pattern, compound
profile, parameter budget, checkpoint revision, normalization, dataset partition
and optimization budget. Keep those conditions shared across comparison nodes.
Use distinct experiments for host normalization and statistics fitted only on
training frames. Evaluation must consume each node's finalized checkpoint using
`checkpoint_from`; it must not reuse a reference adapter accidentally.

The generic [ablation interface](ABLATIONS.md) can enforce declared factor-only
differences. A future shared configuration collection can publish these research
recipes independently of OVLAB and the plugin projects.
