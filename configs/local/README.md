# Local configuration profiles

Machine-specific absolute paths and device assignments belong only in this
directory. Copy `profile.example.yaml` to a descriptive name and edit the copy.
All `*.yaml` profiles except the example are gitignored.

Experiment files never select a local profile. The profile is an explicit
invocation input so the same scientific configuration can be resolved on
different machines without changing versioned files.

Local profiles may contain checkpoint, dataset, and run roots plus logical
device mappings. `resources.checkpoints.<checkpoint-id>.local_path` may point to
an unpublished immutable checkpoint, such as a local QuIC artifact. It is an
execution-only location: the portable registry remains authoritative for the
repository, revision, file sizes, and SHA-256 identity. Local profiles must not
contain credentials or model settings.

The recommended dataset root is outside the checkout at
`ovlab-data/datasets/libero`. Repository-local dataset storage is not part of
the OVLAB deployment contract.

The optional `execution.libero.renderer.device_id` overrides the portable EGL
profile's device selection for one machine. Renderer backend selection belongs
to a reusable profile under `configs/profiles/`, not to the local profile.
