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

## LIBERO renderer execution profiles

Renderer selection is execution-only and is not part of a scientific
experiment document. Select `profiles/libero-bench-egl.yaml` for headless
benchmarking or `profiles/libero-playground-glfw.yaml` for an interactive
playground. The selected profile and resolved renderer are included in the
execution configuration hash; changing profiles leaves the scientific hash,
tasks, seeds, observation/action contracts, and metrics unchanged.

The EGL profile resolves its device from the local profile's optional
`execution.libero.renderer.device_id`, falling back to the index in
`devices.primary_gpu`. GLFW never emits an EGL device setting. Diagnostic
process environment values have precedence: `MUJOCO_GL` selects `egl` or
`glfw`, and `MUJOCO_EGL_DEVICE_ID` overrides the EGL device. Unsupported values
fail during configuration resolution. No MuJoCo, Robosuite, or LIBERO import is
performed by the resolver.

Machine paths and devices belong in gitignored `local/*.yaml` profiles. The
resolver produces a scientific hash without that profile and an execution hash
including all resolved paths and devices. Its immutable output is one
`resolved_config.yaml`.

## Checkpoint resources

Portable policy configurations select a logical `settings.checkpoint_id`.
`resources/registry.yaml` binds that ID to its repository, immutable `revision`,
aggregate SHA-256, and file manifest. Absolute paths never belong in an
experiment or the portable registry.

A gitignored local profile may provide
`resources.checkpoints.<checkpoint-id>.local_path` for an unpublished artifact
such as a QuIC checkpoint. The path is execution-only. Repository, revision,
file sizes, and hashes remain registry-controlled and therefore remain in the
scientific configuration hash.

Docker deployment resolves global Hugging Face cache, local override, and
OVLAB-managed cache in that order. The policy always sees the selected artifact
at `/checkpoints/resolved/<checkpoint-id>` and cannot observe its host location.

Only OpenVLA Vanilla currently has a complete policy configuration. LoRA, OFT,
and QuIC experiment files will be added together with their owner settings and
adapters rather than being represented by non-runnable placeholders.
