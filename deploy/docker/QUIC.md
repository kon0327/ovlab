# QuIC OpenVLA runtime

Build the dedicated serving and training roles from the repository root:

```bash
bash deploy/scripts/build-images.sh policy-openvla-quic training-openvla-quic
```

The images are `ovlab-policy-openvla-quic:local` and
`ovlab-training-openvla-quic:local`. They install the portable
`code/packages/quic-petf-vla` method before the host integration in
`external/openvla-quic`. Neither image installs LIBERO, Robosuite, MuJoCo, or
OpenGL. OVLAB's benchmark image owns rollout environments and communicates
with the policy over the existing AF_UNIX service contract.

Both roles use `deploy/locks/quic-openvla.pylock.toml` and its generated,
hashed requirements. This closure adapts OVLAB's reviewed training runtime:
Python 3.10, Torch 2.2.0/CUDA 12.1 from digest-pinned images, standard
Transformers 4.40.1, NumPy 1.26.4, FlashAttention 2.5.5, TensorFlow 2.15.0,
and the source migration fixes `tensorflow-metadata==1.17.2` and
`protobuf==4.21.12`. The latter hashes match the source project's
`environments/openvla/requirements-fixes.txt`; wheel coordinates were checked
against [TensorFlow Metadata release metadata](https://pypi.org/pypi/tensorflow-metadata/1.17.2/json)
and [Protobuf release metadata](https://pypi.org/pypi/protobuf/4.21.12/json).
The immutable dlimp commit is built separately; the QuIC method has no dlimp
dependency. TensorFlow remains in serving because the host preprocessing
contract uses its image transforms. The OFT Transformers fork is excluded.
This is a reconstructed OVLAB runtime, not an exact copy of the source
project's Conda snapshot. The closure also pins `ninja==1.13.0` for
FlashAttention and `pyopenssl==26.3.0` for compatibility with the inherited
`cryptography==49.0.0`; the older base-image pyOpenSSL is incompatible.
Each image runs `pip check` after installation
and records its actual Pip/Conda inventories under `/usr/share/ovlab/`.

Regenerate the requirements after an intentional lock update:

```bash
python deploy/scripts/render_requirements_lock.py \
  deploy/locks/quic-openvla.pylock.toml deploy/locks/quic-openvla.requirements.txt
```

Use the existing `ovlab deploy run` command with an experiment declaring
`deployment: {profile: quic, renderer: egl}` and policy type `openvla_quic`.
The composed policy's `settings.quic.adapter_path` selects the adapter
directory; a relative path is interpreted from the OVLAB repository root.
`OVLAB_QUIC_ADAPTER_HOST_PATH` in the Compose environment can override its
machine-local location. Relative overrides resolve from the environment
file directory. The deployment verifies `adapter.json` and
`adapter.safetensors`, mounts that directory read-only at `/adapters/quic`,
and sets `OVLAB_QUIC_ADAPTER_PATH` for the provider. The dense host checkpoint
uses the existing verified, read-only checkpoint resolver. Simulator and
benchmark processes receive no adapter mount.

Override images through `OVLAB_QUIC_IMAGE` and `OVLAB_TRAINING_QUIC_IMAGE`.
The existing source, dependency-lock, Dockerfile, checkpoint, configuration,
and deployment provenance contracts apply. Source manifests additionally
hash local changes inside external submodules, so distinct dirty adapter
implementations cannot share an identity merely because they have the same
submodule commit.

# Console progress

`ovlab_core.progress.ConsoleProgress` provides shared stderr tracking for
optimizer steps, validation batches, and evaluation episodes. Interactive
terminals receive an updating line; captured Docker logs receive a start
line, progress at most every 30 seconds, and the final actual count and
status. Failure or interruption preserves the partial count. `OVLAB_PROGRESS=on`
prints every update in captured logs; `OVLAB_PROGRESS=off` silences tracking.
JSON command results remain on stdout. Console progress supplements persisted
scientific metrics and does not change their values.

```python
from ovlab_core.progress import ConsoleProgress

with ConsoleProgress("Train", total=max_steps, unit="step") as progress:
    for step in range(max_steps):
        loss = optimize_one_step()
        progress.advance(detail=f"loss={loss:.4f}")
```

The upstream OpenVLA trainer already uses tqdm. OVLAB's rollout runner now
tracks completed episodes for every benchmark; QuIC training uses the shared
helper for optimizer steps and teacher-forced validation.
