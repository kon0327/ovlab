# ovlab-benchctl

This package is OVLAB's configuration composition root. It deliberately sits
above concrete benchmark, policy, metric, runner, and artifact packages. Those
owner packages remain independently importable and do not depend on this
resolver.

Configuration uses a strict dependency-free YAML subset, explicit relative
`extends`, exact per-kind schemas, and explicit experiment component
references. Duplicate keys, unknown keys, implicit merge keys, anchors,
aliases, tags, multi-document files, non-two-space indentation, reference
traversal, and inheritance cycles are rejected. Mapping inheritance is a deep
merge; sequences and scalars replace their parent value.

`ConfigResolver.resolve()` validates and composes all component documents,
resolves logical checkpoint/device/artifact resources through an explicitly
selected local profile, constructs the existing immutable owner settings, and
checks the shared action and observation interfaces. It returns separate
scientific and execution hashes. The scientific hash excludes the local
profile and resolved machine paths/devices; the execution hash includes them.

`ResolvedExperimentConfig.write()` creates one deterministic
`resolved_config.yaml` and refuses to overwrite an existing file. Runner CLI
construction and experiment execution are intentionally deferred.
