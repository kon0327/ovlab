#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_paths=("$repository_root/code/apps/benchctl/src")
for source_path in "$repository_root"/code/packages/*/src; do
    [[ -d "$source_path" ]] && source_paths+=("$source_path")
done
joined_paths="$(IFS=:; echo "${source_paths[*]}")"
export PYTHONPATH="$joined_paths${PYTHONPATH:+:$PYTHONPATH}"
exec "${OVLAB_PYTHON:-python3}" "$repository_root/deploy/scripts/build_images.py" --root "$repository_root" "$@"
