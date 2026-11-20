# ovlab-policy-sdk

`ovlab-policy-sdk` defines the synchronous policy boundary used by local and
future remote policy implementations. A policy declares its required named
observations, output `ActionSpec`, supported action horizons, instruction
behavior, deterministic-reset support, and whether raw model output is retained.

Policies use the lifecycle `created → ready → episode_active → ready → closed`.
The runner calls `end_episode()` after benchmark termination or truncation;
`close()` is idempotent. `PolicyAdapter.predict()` validates that every returned
`ActionPrediction` matches the declared action specification and horizon.

The SDK consumes only `PolicyObservation`. Evaluation-only and privileged
signals are deliberately absent from this interface. Model loading, transport,
checkpoint selection, decoding, and benchmark-specific conversions belong to
concrete policy integrations, not this dependency-light package.

Both adapter packages depend independently on `ovlab-core`; neither adapter
package depends on the other.

## Model plugins

The versioned `ovlab_policy_sdk.plugin.PolicyPlugin` API connects external model
projects to configuration, Docker builds, serving and evaluation. See the
[developer guide](../../../docs/POLICY_PLUGINS.md) and
[standalone example](../../../examples/policy-plugin/).

Start with the example README for the complete CPU service workflow. Plugin control
hooks are stateless and run on the host; model and episode state belong to the
adapter. The settings returned by `resolve_settings` must expose the structural
`PolicySettings.record_raw_output: bool` contract.


Training and checkpoint preparation extensions are documented in [TRAINING.md](TRAINING.md).

`ovlab_policy_sdk.strict_yaml` provides the same strict configuration parser used
by OVLAB, without importing its orchestration package. Plugins can use `load`,
`loads` and `dumps`; parsing failures raise `PolicyPluginValidationError`.
