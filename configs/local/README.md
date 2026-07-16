# Local configuration profiles

Machine-specific absolute paths and device assignments belong only in this
directory. Copy `profile.example.yaml` to a descriptive name and edit the copy.
All `*.yaml` profiles except the example are gitignored.

Experiment files never select a local profile. The profile is an explicit
invocation input so the same scientific configuration can be resolved on
different machines without changing versioned files.

Local profiles may contain checkpoint, dataset, and run roots plus logical
device mappings. They must not contain credentials or model settings.
