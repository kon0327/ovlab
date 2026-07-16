# OVLAB configuration

OVLAB composes experiments from explicit, versioned component documents. There
is no implicit search path, Hydra defaults list, environment-variable expansion,
or runtime mutation. An experiment names every component and the resource
registry; a machine-local profile is selected separately by the caller.

Inheritance is opt-in through a relative `extends` path. Mappings deep-merge,
while scalar and sequence values replace the parent. Every composed document is
validated against an exact schema, so misspelled or unknown keys fail early.

The shared LIBERO action interface records the already verified
`closed_positive` gripper convention. Both benchmark and policy references must
resolve to that same interface, and the resolver compares it with the concrete
LIBERO adapter contract. Canonical camera names are checked in the same way.

Machine paths and devices belong in gitignored `local/*.yaml` profiles. The
resolver produces a scientific hash without that profile and an execution hash
including all resolved paths and devices. Its immutable output is one
`resolved_config.yaml`.

Only OpenVLA Vanilla currently has a complete policy configuration. LoRA, OFT,
and QuIC experiment files will be added together with their owner settings and
adapters rather than being represented by non-runnable placeholders.
