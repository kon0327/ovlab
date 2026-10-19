# External repositories

OVLAB normally attaches external repositories as Git submodules. A separately
identified user-supplied source archive may be preserved as a content-addressed
snapshot when no authoritative Git revision exists.

| Component | Local path | Source type | Repository URL | Role within OVLAB | Consumers |
| --- | --- | --- | --- | --- | --- |
| LIBERO | `external/libero` | upstream | <https://github.com/Lifelong-Robot-Learning/LIBERO.git> | Primary simulation benchmark | OVLAB runner and benchmark adapter |
| OpenVLA | `external/openvla` | upstream | <https://github.com/openvla/openvla.git> | Shared upstream policy source | Vanilla and LoRA policy services |
| OpenVLA-OFT | `external/openvla-oft` | upstream | <https://github.com/moojink/openvla-oft.git> | OFT implementation with an independent dependency stack | OFT policy service |
| OpenVLA-QuIC | `external/openvla-quic` | fork | <https://github.com/kon0327/openvla-quic.git> | Dedicated fork containing QuIC architectural modifications | QuIC policy service |
| quic-peft-vla | `external/quic-peft-vla` | standalone Python Git submodule, MIT | <https://github.com/kon0327/quic-peft-vla.git> | Portable QuIC adapters extracted from the supplied `quic-petf-mpv` source | OpenVLA-QuIC and other dense PyTorch hosts |
| compound-PEFT | `external/compound-peft` | user-supplied archive | unavailable | Immutable legacy technical reference; not official or OpenVLA-validated | QuIC-PEFT backend bridge only |

Exact revisions are pinned by the root repository's Git submodule gitlinks rather than duplicated in documentation. Inspect them with:

```bash
git submodule status --recursive
```

The compound-PEFT snapshot is not a submodule and must not be represented as a
Git revision. Its provenance is recorded in `external/compound-peft.provenance.yaml`
and its sorted content manifest in `external/compound-peft.manifest.sha256`.

`quic-peft-vla` is an independent Git repository pinned as a submodule. Its
source distribution retains its existing Python distribution/import names for
compatibility. Its tests, CI, license, and standalone installation instructions
live in [`quic-peft-vla/`](quic-peft-vla/README.md). Its `PROVENANCE.json`
records method attribution and license decisions. It depends only on Torch and
safetensors.
OpenVLA-QuIC declares it as a versioned dependency; OVLAB's QuIC images install
this external package before installing the OpenVLA fork.
