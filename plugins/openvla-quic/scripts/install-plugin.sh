#!/usr/bin/env bash
set -euo pipefail

plugin_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --no-cache-dir --no-deps --no-build-isolation "$plugin_root"
