# Historical reference sources

`compound-peft/` is the unchanged user-supplied historical source archive retained
for technical analysis. Its original sorted content manifest is
`compound-peft.manifest.sha256`; source attribution and limitations are recorded
in `compound-peft.provenance.yaml`. It is not an official or OpenVLA-validated
implementation and is not used as a standalone runtime model.

Runtime sources are selected by the [plugin registry](../docs/POLICY_PLUGINS.md)
and materialized into ignored `external/`. Their Git URLs and immutable commits
are recorded in the registry and source manifests. The portable QuIC runtime
uses registered `openvla-quic` and `quic-peft-vla` projects.
