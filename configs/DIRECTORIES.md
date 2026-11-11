# Configuration directories

This guide maps the configuration tree and explains where to find each kind of
configuration. Paths below are relative to `configs/`. For composition rules,
resource resolution and runnable examples, see [README.md](README.md).

## Directory tree

```text
configs/
├── ablations/                 Training ablation suites
├── artifacts/                 Artifact store definitions
├── benchmarks/                Benchmark components
│   ├── libero/                LIBERO suites and variants
│   ├── mock/                  Lightweight mock benchmark
│   └── playground/            Placeholder for playground configurations
├── deploy/                    Policy service deployment scenarios
├── exports/                   Legacy grouped export specifications
├── interfaces/
│   └── actions/               Policy/benchmark action contracts
├── local/                     Machine-specific resource profiles
├── metrics/                   Evaluation metric sets
├── policies/                  Policy components
│   ├── mock/                  Mock policies
│   ├── openvla-lora/          Merged OpenVLA LoRA policies
│   ├── openvla-oft/           OpenVLA-OFT policies
│   ├── openvla-quic/          OpenVLA QuIC policies
│   └── openvla-vanilla/       Base OpenVLA policies
├── profiles/                  Reusable execution profiles
├── protocols/                 Evaluation execution and recording protocols
└── resources/                 Portable resource registry
```

Directories containing only `.gitkeep`, such as `benchmarks/playground/`, are
placeholders rather than populated configuration families.

## Scenarios and orchestration

Training, evaluation and workflow scenarios live in the separate
[`experiments/`](../experiments/README.md) directory.

- **`ablations/`** defines training ablation suites with a common base profile,
  varied factors and variant profiles. See the [ablation guide](../docs/ABLATIONS.md).
- **`deploy/`** contains policy-only deployment scenarios, including the policy,
  registry, runtime and service settings. Use `ovlab deploy run TARGET`; see
  [deployment documentation](../DEPLOYMENT.md).

## Reusable components

- **`benchmarks/`** defines environments, task selection, observations and
  action-interface references. `libero/` contains LIBERO suite and smoke
  variants; `mock/` provides a small test benchmark; `playground/` is currently
  a placeholder.
- **`policies/`** defines model inference settings, inputs, checkpoint IDs,
  runtime options and action interfaces. Its subdirectories separate mock,
  merged LoRA, OFT, QuIC and vanilla OpenVLA policies; some include quantized
  variants.
- **`interfaces/actions/`** defines shared action contracts, such as action
  dimensions, representation, bounds and gripper convention. A policy and its
  benchmark must reference compatible contracts.
- **`metrics/`** defines metric sets, required metrics and plugin settings for
  evaluation, including action safety and qualification metrics.
- **`protocols/`** defines evaluation execution rules such as rollout counts,
  seeds, episode limits, action-chunk behavior, recording and reproducibility.
- **`artifacts/`** defines artifact storage components. `filesystem.yaml`
  selects the filesystem store using the logical `runs_root` resource.
- **`exports/`** holds legacy `ovlab.export-spec/v1` grouped export
  specifications. New exports use built-in versioned templates; use
  `ovlab report EXPERIMENT_ID` for new grouped comparisons.

## Resources and execution environment

- **`resources/`** holds the portable registry of logical resources, including
  checkpoint identities, revisions and file hashes. Policies refer to logical
  checkpoint IDs rather than machine-specific paths.
- **`profiles/`** holds reusable execution profiles. The LIBERO EGL profile is
  for headless benchmarking; the GLFW profile is for interactive rendering.
- **`local/`** holds machine-specific paths and device assignments. Copy
  `profile.example.yaml` to create a local profile, and select it separately
  when invoking a command. Local YAML profiles other than the example are
  gitignored. See [local/README.md](local/README.md).

Evaluation scenarios compose reusable components and the registry; local
profiles supply machine-specific locations and devices separately. This keeps
portable scientific configuration distinct from execution-specific settings.
