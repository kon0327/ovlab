#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir --no-deps --no-build-isolation \
  /opt/ovlab/code/packages/ovlab-core \
  /opt/ovlab/code/packages/ovlab-policy-sdk \
  /opt/ovlab/code/packages/ovlab-benchmarks \
  /opt/ovlab/code/packages/ovlab-metrics \
  /opt/ovlab/code/packages/ovlab-openvla-common \
  /opt/ovlab/code/packages/ovlab-remote-policy \
  /opt/ovlab/code/apps/runner \
  /opt/ovlab/code/policies/openvla-vanilla \
  /opt/ovlab/code/policies/openvla-lora-merged \
  /opt/ovlab/code/policies/openvla-oft \
  /opt/ovlab/code/policies/openvla-quic \
  /opt/ovlab/code/apps/benchctl
