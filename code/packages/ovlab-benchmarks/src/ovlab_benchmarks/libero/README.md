# LIBERO benchmark adapter

This integration targets the pinned LIBERO source at commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. Production imports are lazy:
`import ovlab_benchmarks` and the generic interfaces do not require LIBERO,
Robosuite, MuJoCo, Torch, or EGL. Constructing the default concrete adapter
requires the tested local LIBERO installation and its existing initial-state
assets; it never downloads them.

## Suites and deterministic tasks

Supported stable suite names are `LIBERO-Spatial`, `LIBERO-Object`,
`LIBERO-Goal`, and `LIBERO-10`, mapped to the pinned registry identifiers
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. Native task
order zero is preserved. Stable IDs use `libero/<suite-slug>/<native-index>`,
for example `libero/spatial/0`; absolute paths are not identifiers.

`ROLLOUT_INDEX` selects `rollout_index % initial_state_count`. `SEEDED` uses a
local NumPy generator seeded from the adapter base seed, episode seed, and
rollout index. Neither strategy touches NumPy's global RNG. The default native
environment seed remains `0`, matching the verified evaluation scripts.

After `reset()` and `set_init_state()`, the default ten settling steps execute
`[0, 0, 0, 0, 0, 0, -1]`. They do not create policy predictions or increment
the policy step index. Selection and settling details are recorded in reset
metadata.

## Observation profiles

- `PRIMARY_RGB`: third-person `agentview` RGB only.
- `DUAL_RGB`: `agentview` and `robot0_eye_in_hand` RGB.
- `RGB_PROPRIOCEPTION`: primary RGB plus verified EEF position, EEF XYZW
  quaternion, and two gripper joint positions.

Images remain at the configured simulator resolution in HWC `uint8` RGB. Both
axes are reversed, producing an explicit 180-degree rotation. This matches the
pinned OpenVLA and OpenVLA-OFT evaluation utilities. The adapter does not resize,
JPEG-roundtrip, normalize, or create model tensors.

| Native LIBERO data | OVLAB name | Access | Shape | Units | Notes |
|---|---|---|---|---|---|
| `agentview_image` | `camera.primary.rgb` | policy | H×W×3 | uint8 RGB | Rotated 180° |
| `robot0_eye_in_hand_image` | `camera.wrist.rgb` | policy | H×W×3 | uint8 RGB | Dual profile; rotated 180° |
| `robot0_eef_pos` | `robot.eef.position` | privileged or policy | 3 | m | Policy-visible only in proprio profile |
| `robot0_eef_quat` | `robot.eef.orientation_xyzw` | privileged or policy | 4 | unitless | Native XYZW quaternion |
| `robot0_gripper_qpos` | `robot.gripper.joint_position` | privileged or policy | 2 | rad | Native gripper joint state |
| `check_success()` | `benchmark.task_success` | evaluation | scalar | boolean | Authoritative goal predicate |
| step reward | `benchmark.reward` | evaluation | scalar | native | Not used as success |
| OVLAB state | `episode.terminated` | evaluation | scalar | boolean | Success or native terminal |
| OVLAB limit | `episode.truncated` | evaluation | scalar | boolean | Configured policy-step limit |
| adapter counter | `episode.native_step_index` | evaluation | scalar | count | Excludes settling steps |
| selected state | `episode.initial_state_index` | evaluation | scalar | index | Deterministic selection |

Privileged robot signals are not duplicated when the proprioception profile
makes them policy-visible. Object-specific signals are not declared because the
current run-wide `SignalRegistry` cannot express task-specific schemas. Raw
contacts and a generic `collision_event` are deliberately unavailable: native
contacts do not distinguish intended manipulation from harmful collision.

## Action convention and termination

The accepted command is the pinned default `OSC_POSE` input with seven
normalized dimensions in `[-1, 1]`:

```text
[delta_position_xyz(3), delta_axis_angle_xyz(3), gripper(1)]
```

All dimensions are normalized controller commands, not direct metres or
radians. Gripper convention is closed-positive: `-1` opens and `+1` closes.
This is the command accepted by `env.step()`. OpenVLA-specific conversion from
`[0,1]`, binarization, and sign inversion belongs to its policy adapter and is
not performed here. Commands are neither reshaped nor clipped. `applied_action`
records the exact value handed to `env.step()`; any internal Robosuite actuator
clipping is not observable through this API.

The settings contract reserves a controller override field, but the pinned
`OffScreenRenderEnv` wrapper fixes `OSC_POSE` internally and cannot accept an
override safely. A non-null override therefore fails explicitly instead of
claiming unsupported action semantics.

Task success comes from `check_success()`. Native `done` or success produces
termination; reaching the configured OVLAB policy-step limit without success
produces truncation. Native reward remains separate.

## Rendering and manual validation

Rendering is headless and contained inside the integration. WSL/EGL variables
such as `MUJOCO_GL=egl` and the desired visible GPU must be configured before
Python starts. OVLAB does not overwrite them or silently select another GPU.
An explicit `render_gpu_device_id` is forwarded only when configured.

Run dependency-light tests with `ovlab-tester`. Run the real smoke test only in
the already verified LIBERO environment, with local assets available:

```bash
OVLAB_RUN_LIBERO=1 MUJOCO_GL=egl \
  conda run -n openvla deploy/scripts/test.sh \
  code/tests/integration/benchmarks/libero/test_real_libero.py -q -s
```

The real test performs one deterministic reset and one safe open-gripper no-op
step. It does not load a policy, checkpoint, demonstration, or full suite.
