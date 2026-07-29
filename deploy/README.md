# Reproducible OVLAB deployment

This directory packages the accepted OVLAB runtime boundaries without changing
the host Conda environments. Production containers expose only the `ovlab` CLI.
They communicate through a versioned AF_UNIX socket in `/run/ovlab`; no TCP
port, host networking, Docker socket, privileged mode, or source-tree bind is
used.

## Production image matrix

| Image | Runtime responsibility | Included heavy closure | Deliberately absent |
|---|---|---|---|
| `ovlab-benchmark-libero` | CLI, runner, LIBERO, Robosuite, MuJoCo, EGL and immutable trace output | LIBERO simulation stack; Torch only because pinned LIBERO imports it to load initial states | OpenVLA, Transformers, PEFT, FlashAttention and model weights |
| `ovlab-policy-openvla` | Vanilla and merged-LoRA OpenVLA service | Torch 2.2.0/CUDA 12.1, Transformers 4.40.1, PEFT 0.11.1, FlashAttention 2.5.5 and pinned OpenVLA source | LIBERO, Robosuite, MuJoCo, datasets and run output |
| `ovlab-policy-openvla-oft` | OpenVLA-OFT service | The distinct OFT Transformers commit, Torch/CUDA, FlashAttention and pinned OFT source | LIBERO, Robosuite, MuJoCo, datasets and run output |

QuIC remains descriptor-only. There is no QuIC image, Compose profile, provider
selection, or claim of runtime readiness in Gate H.

## Reproducibility and provenance

Base images are pinned by digest in `docker/base-images.lock`. Python closures
are resolved into PEP 751 `locks/*.pylock.toml` files with artifact SHA-256
hashes. Docker consumes generated `--require-hashes` requirement files with
dependency resolution disabled. The OFT Transformers fork is the single VCS
artifact; both requested revision and resolved commit are
`bc339d9ad707454c0c115970db43c260067c61ab`.

`scripts/source_manifest.py` records HEAD, clean/dirty state, submodule gitlinks,
untracked build-relevant files, and content hashes deterministically. The build
script embeds that manifest and adds OCI labels for repository revision, source
hash, dirty state, lock hash, Dockerfile hash and image role. Dirty builds are
allowed but never presented as clean.

Build all mandatory production images:

```bash
bash deploy/scripts/build-images.sh
```

The host needs Docker Engine with Compose v2, NVIDIA Container Toolkit, a driver
compatible with CUDA 12.1, and enough local storage for the pinned images. Verify
these prerequisites with `docker version`, `docker compose version`, `docker info`,
`nvidia-smi`, and a small `docker run --gpus all` check.

For a deployment, set the three `OVLAB_*_IMAGE` Compose variables to immutable
registry digest references and set the matching `*_IMAGE_DIGEST` values. Those
identities enter execution provenance and the policy handshake, but never the
scientific configuration hash. Machine paths, socket names, renderer device IDs
and timing also remain outside the scientific hash.

Copy `compose/.env.example` to an untracked `.env` and set local paths. Record the
deployment identity before launch, for example:

```bash
export OVLAB_DEPLOYMENT_MANIFEST_SHA256="$(sha256sum deploy/compose/compose.yaml | cut -d' ' -f1)"
export OVLAB_CONTAINER_RUNTIME_VERSION="$(docker version --format '{{.Server.Version}}')"
```

## Runtime mounts and permissions

All containers run as UID/GID `10001:10001`, drop all capabilities, enable
`no-new-privileges`, use a read-only root filesystem and have no network. The
policy receives only `/checkpoints:ro` and the shared socket volume. The
benchmark receives `/datasets:ro`, the socket volume and `/runs` as its only
persistent writable mount. Configuration is baked into each image.

Override the portable repository-relative asset locations when necessary:

```bash
export OVLAB_CHECKPOINTS_PATH=/host/read-only/huggingface
export OVLAB_DATASETS_PATH=/host/read-only/libero
export OVLAB_RUNS_PATH=/host/ovlab-runs
export OVLAB_GPU_DEVICE=0
export OVLAB_EGL_DEVICE_ID=0
```

These are deployment settings, not scientific parameters. Do not put secrets in
them. Hugging Face and Transformers offline modes are forced in policy services.

## Compose profiles

The accepted OpenVLA topology is:

```bash
docker compose --file deploy/compose/compose.yaml --profile openvla up \
  --abort-on-container-exit --exit-code-from benchmark-openvla
```

The separate OFT topology is selected with `--profile oft`. The benchmark waits
for a protocol-aware health probe. Readiness is advertised only after provider
initialization; the health operation performs neither prediction nor trace
creation and does not consume the service connection.

GPU access uses Compose `gpus: all` plus `NVIDIA_VISIBLE_DEVICES`; set a concrete
device in reproducible deployments. The benchmark forces EGL. GLFW remains an
interactive, dependency-light-tested playground path and is not required by
headless deployment.

The containers preserve the Gate G workflow through the same public entrypoint:
use `ovlab config validate ...` or `ovlab config resolve ...`, start the selected
foreground `ovlab service serve ...`, use `ovlab connect ...` for a no-prediction
capability probe, and then `ovlab run ...`. Add `--json` where supported for stable
machine output; omit it for human output. Compose delegates directly to the service
and run commands and returns their exit status.

Use `docker compose ... down` for scoped teardown. Do not use global prune commands.
On a signal, the foreground CLI closes its adapter and removes only its owned socket;
partial run artifacts already created remain available for inspection.

## Test-only transport smoke

The mock provider exists only in `Dockerfile.smoke`; no production image copies
it. It validates two non-root containers, the shared socket, health ordering,
capability negotiation and cleanup without loading a model or LIBERO:

```bash
bash deploy/scripts/transport-smoke.sh
```

This smoke is evidence for packaging and IPC only. It is not scientific model or
benchmark evidence.

## Troubleshooting and limitations

- A missing checkpoint or dataset is an expected hard failure: verify the read-only
  host binding and registry identity; runtime networking is disabled and will not
  repair it by downloading.
- For `cuda is not available`, verify the NVIDIA runtime, driver compatibility and
  selected `OVLAB_GPU_DEVICE`.
- For EGL failures, validate `MUJOCO_EGL_DEVICE_ID`, graphics driver exposure and the
  detected renderer. WSL may expose Mesa `llvmpipe`; this is valid headless EGL but
  is software rendering, not NVIDIA renderer evidence.
- Socket permission failures usually mean the shared volume was not created by the
  Compose project or the fixed UID/GID contract was overridden inconsistently.
- The default container UID/GID is `10001:10001`; ensure the writable runs directory
  accepts that identity without making it world-writable.

Gate H validates packaging, isolation, IPC, CUDA visibility and headless startup. It
does not validate model output, task success, latency, full evaluation, QuIC math or
deployment safety. QuIC-PEFT and QuIC-WC remain descriptor-only and non-runnable.
