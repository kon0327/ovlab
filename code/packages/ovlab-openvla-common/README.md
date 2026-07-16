# OVLAB OpenVLA Common

`ovlab-openvla-common` contains the immutable, NumPy-only contracts shared by
OVLAB OpenVLA policies. It depends only on `ovlab-core` and NumPy. It must not
import Torch, Transformers, OpenVLA, LIBERO, PEFT, OFT, or QuIC.

The package owns the versioned prompt formatter, canonical RGB validation,
checkpoint-identity contract, and the target codec shared by future OpenVLA
variants. It does not load models or checkpoints.

## Verified behavior

The implementation follows OpenVLA commit
`c8f03f48af692657d3060c19588038c7220e9af9`. The standard prompt template is
identified as `openvla-v1@1.0.0` and is exactly:

```text
In: What action should the robot take to {instruction.lower()}?
Out:
```

The decoded OpenVLA pose values are left unchanged. The decoded gripper value
uses `0 = closed, 1 = open`; the `openvla-decoded-to-libero-osc-pose@1.0.0`
codec applies `-sign(2*g - 1)`. Its output therefore uses `+1 = closed,
-1 = open`, matching LIBERO's normalized OSC_POSE command. At the exact `0.5`
boundary the pinned upstream NumPy operation yields `0`. A typed pre-codec
value prevents accidental application of this conversion twice.

Run its dependency-light coverage through:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/unit/policies/openvla_vanilla \
  code/tests/contract/policies/openvla_vanilla
```
