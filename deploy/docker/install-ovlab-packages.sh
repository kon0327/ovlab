#!/usr/bin/env bash
set -euo pipefail
packages=()
for project in /opt/ovlab/code/packages/* /opt/ovlab/code/policies/* /opt/ovlab/code/apps/*; do
  if [[ -f "$project/pyproject.toml" ]]; then packages+=("$project"); fi
done
python -m pip install --no-cache-dir --no-deps --no-build-isolation "${packages[@]}"
