#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repository_root/code/apps/benchctl/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${OVLAB_PYTHON:-python3}" "$repository_root/deploy/scripts/build_images.py" --root "$repository_root" "$@"
