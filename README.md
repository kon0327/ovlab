# OpenVLABenchmark (OVLAB)

OpenVLABenchmark (OVLAB) is a reproducible experimental framework for evaluating OpenVLA-derived Vision–Language–Action policies, initially with the LIBERO benchmark.

OVLAB follows a **Config → Connect → Run** workflow: define an experiment, connect interchangeable policies and benchmarks through shared contracts, and execute reproducible runs whose traces can later be recorded, analyzed, and exported.

The experiment runner and benchmark adapter execute together in a runner process or container. Each VLA implementation runs independently as a policy service, communicating through a dependency-light policy protocol. This separation allows Vanilla, LoRA, OFT, and QuIC implementations to keep distinct dependency environments.

## Repository layout

- `code/`: core packages, the policy SDK, benchmark adapters, metrics, policy integrations, applications, and tests.
- `configs/`: benchmark, policy, metric, protocol, and experiment configuration.
- `deploy/`: reserved for Docker, Compose, and deployment scripts.
- `external/`: destinations for pinned external repositories and the dedicated OpenVLA-QuIC fork.
- `checkpoints/`: local model checkpoints; generated contents are not versioned.
- `datasets/`: local benchmark datasets; generated contents are not versioned.
- `runs/`: generated experiment manifests, traces, and results; contents are not versioned.

OVLAB is currently in the **scaffolding phase**. Package implementations, dependency definitions, protocols, containers, and runtime configuration will be designed in later phases.

## Testing

OVLAB uses the lightweight `ovlab-tester` Conda environment for CPU-only automated tests. Create it from `deploy/environments/ovlab-tester/environment.yml`, then run:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh
```

GPU and policy-specific smoke tests remain isolated in their corresponding policy environments.
