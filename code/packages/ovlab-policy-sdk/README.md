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
