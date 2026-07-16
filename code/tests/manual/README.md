# Manual tests

This directory contains hardware-dependent smoke tests and proven launch harnesses that are intentionally excluded from the automated unit, integration, and contract test suites.

Manual tests may require a GPU, policy-specific dependencies, model checkpoints, datasets, or network access. Run them explicitly in the matching policy-service environment; repository validation must not execute them automatically or download their inputs.
