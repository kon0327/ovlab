#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

source_paths=()
for source_path in code/packages/*/src code/policies/*/src; do
    if [[ -d "$source_path" ]]; then
        source_paths+=("$repository_root/$source_path")
    fi
done

if [[ ${#source_paths[@]} -eq 0 ]]; then
    echo "No OVLAB src-layout packages were found." >&2
    exit 1
fi

joined_paths="$(IFS=:; echo "${source_paths[*]}")"
export PYTHONPATH="$joined_paths:$repository_root/code/tests"

if [[ $# -eq 0 ]]; then
    set -- code/tests/unit
fi

exec python -m pytest "$@"
