#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$repository_root/deploy/scripts/train-quic-lora-coverage.py" "$@"
