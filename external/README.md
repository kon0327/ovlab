# External repositories

OVLAB attaches its external source dependencies as Git submodules. These repositories retain their independent histories and dependency stacks; OVLAB does not vendor or modify their source code.

| Component | Local path | Source type | Repository URL | Role within OVLAB | Consumers |
| --- | --- | --- | --- | --- | --- |
| LIBERO | `external/libero` | upstream | <https://github.com/Lifelong-Robot-Learning/LIBERO.git> | Primary simulation benchmark | OVLAB runner and benchmark adapter |
| OpenVLA | `external/openvla` | upstream | <https://github.com/openvla/openvla.git> | Shared upstream policy source | Vanilla and LoRA policy services |
| OpenVLA-OFT | `external/openvla-oft` | upstream | <https://github.com/moojink/openvla-oft.git> | OFT implementation with an independent dependency stack | OFT policy service |
| OpenVLA-QuIC | `external/openvla-quic` | fork | <https://github.com/kon0327/openvla-quic.git> | Dedicated fork containing QuIC architectural modifications | QuIC policy service |

Exact revisions are pinned by the root repository's Git submodule gitlinks rather than duplicated in documentation. Inspect them with:

```bash
git submodule status --recursive
```
