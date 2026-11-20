# OpenVLA integration for OVLAB

This project supplies OpenVLA Vanilla, merged LoRA, OFT and QuIC policy adapters,
training backends, checkpoint preparation, model configurations and reproducible
image recipes. It is a standalone Python project registered with OVLAB through
`plugin.json`. OVLAB core does not depend on its distribution or model namespaces.

The checked-in `plugins/openvla/` directory is the local reference implementation.
The same project can live in a separate Git repository: register its URL and a
full commit, and OVLAB caches it under ignored `plugins/.cache/` and prepares
`external/<project-source-id>/` inside the temporary Docker context. Local
registrations use a content-identified snapshot of the project. Source acquisition
and image construction are host operations; prediction runs inside its own image.

## Project layout

| Path | Responsibility |
| --- | --- |
| `plugin.json` | Public integration manifest, policy variants, upstream pins and build roles |
| `pyproject.toml` | The single `ovlab-openvla-plugin` distribution |
| `src/ovlab_openvla_common/` | Lightweight OpenVLA configuration, codecs and plugin hooks |
| `src/ovlab_openvla_vanilla/` | Vanilla model adapter and lazy runtime loader |
| `src/ovlab_openvla_lora_merged/` | Merged LoRA policy adapter |
| `src/ovlab_openvla_oft/` | OFT policy adapter and artifact checks |
| `src/ovlab_openvla_quic/` | QuIC policy adapter, descriptor and runtime integration |
| `src/ovlab_openvla_training/` | Training and checkpoint preparation backends |
| `configs/` | Plugin-owned policies, checkpoints and deployment configurations |
| `docker/` | Policy and training images |
| `locks/` | Input requirements and immutable resolved runtime closures |
| `scripts/install-plugin.sh` | Installs this selected project without resolving extra dependencies |

## Connection to OVLAB

The project manifest declares the supported policy `type` values and their
`module:object` entrypoints. Each entrypoint implements `PolicyPlugin` from
`ovlab_policy_sdk.plugin`; the SDK is the integration contract. Core orchestration
loads the selected plugin for configuration validation and settings resolution,
then calls its adapter factory inside the selected policy service. The adapter
implements the SDK lifecycle and the existing observation, action and capability
contracts. Model imports remain lazy so host-side configuration inspection needs
neither Torch nor Transformers.

Policy variants share a distribution while retaining separate runtime images.
Vanilla and merged LoRA share the OpenVLA image. OFT uses its own dependency stack.
QuIC installs its pinned portable adapter package before the pinned OpenVLA fork.
Checkpoint weights and QuIC adapter artifacts are supplied through read-only
runtime mounts; these artifacts are independent of Python source acquisition.

Every build recipe belongs to this project. Its Dockerfile and lock paths are
relative to this directory in `plugin.json`; OVLAB rebases them to the prepared
project snapshot. Docker `COPY` inputs use the managed source names, including
`external/plugin-openvla`. A project registered under another source identifier
must adapt that path in its own Dockerfiles. A new model integration can choose
its own source IDs and recipes without changing core build dispatch.

The recipes first install OVLAB core packages with the core installer and then
install only this plugin project. Infrastructure images retain the core package
set. The source manifest and expanded registry are injected as BuildKit secrets
and saved as build metadata; the expanded registry supplied to an image points
at prepared container paths rather than a developer's local checkout.

## Developing a variant

1. Add a plugin implementation under `src/` using the public SDK contract. Keep
   heavy model imports inside runtime loading and prediction methods.
2. Declare the variant in `plugin.json`, including its entrypoint, required source
   IDs, deployment metadata and build role. Add upstream Git URLs with full commit
   IDs when the implementation needs additional projects.
3. Add policy configuration and checkpoint metadata under `configs/`. Document
   the expected cameras, proprioception, normalization and action representation.
4. Add or reuse a Docker recipe and its dependency locks. The recipe installs core
   plus the selected plugin project with dependency resolution disabled.
5. Exercise validation, settings resolution and observation/action compatibility
   with a small SDK adapter test, then qualify the real model in its runtime image.

Use the generic `ovlab plugin` commands to inspect registration, prepare sources
and build a role. The host OVLAB environment must contain the SDK and core CLI.
For direct Python development, install this project into that environment with
`python -m pip install --no-deps -e plugins/openvla`; production image installation
uses a frozen source snapshot instead of an editable package.

## Dependency locks

The `.in` files select top-level dependencies. Corresponding `.pylock.toml` files
record immutable artifact URLs and SHA-256 hashes; `.requirements.txt` files are
consumed with `--require-hashes`. Locks target CPython 3.10 on Linux x86-64.

OFT's Transformers fork is pinned at
`bc339d9ad707454c0c115970db43c260067c61ab`, and its `dlimp` dependency at
`040105d256bd28866cc6620621a3d5f7b6b91b46`. OFT's accepted inference helper imports
parts of the TensorFlow training stack while loading; its runtime lock includes
that closure. Its TensorFlow Metadata 1.15.0 and W&B 0.16.6 pins retain the Protobuf
range supported by TensorFlow 2.15.0. QuIC has a separately resolved closure.

Plugin source versions, model weight revisions and the OVLAB SDK API version are
separate identities. Change upstream pins or model artifacts deliberately and
retain their verification metadata when producing experimental results.
