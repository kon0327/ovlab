#!/usr/bin/env bash
# Source this script to leave the installed environment active in this shell.
# Avoid changing shell options: they would persist in the caller after `source`.
_ovlab_env_setup() {
    local repository_root bootstrap_python prefix prefix_supplied argument index conda_hook manager requested_manager target
    repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    bootstrap_python="${OVLAB_BOOTSTRAP_PYTHON:-python3}"
    if ! command -v "$bootstrap_python" >/dev/null 2>&1; then
        echo "Python 3 is required to bootstrap the OVLAB environment." >&2
        return 127
    fi

    prefix=""
    prefix_supplied=false
    requested_manager="auto"
    for ((index = 1; index <= $#; index++)); do
        argument="${!index}"
        if [[ "$argument" == --prefix ]]; then
            ((index++))
            prefix="${!index:-}"
            prefix_supplied=true
        elif [[ "$argument" == --prefix=* ]]; then
            prefix="${argument#--prefix=}"
            prefix_supplied=true
        elif [[ "$argument" == --manager ]]; then
            ((index++))
            requested_manager="${!index:-}"
        elif [[ "$argument" == --manager=* ]]; then
            requested_manager="${argument#--manager=}"
        fi
    done
    if [[ "$requested_manager" == auto ]]; then
        if command -v conda >/dev/null 2>&1; then manager=conda; else manager=venv; fi
    else
        manager="$requested_manager"
    fi
    if [[ "$prefix_supplied" == true && -z "$prefix" ]]; then
        echo "--prefix requires a path." >&2
        return 2
    fi
    if [[ "$manager" == conda && "$prefix_supplied" == false ]]; then
        target="ovlab-env"
    else
        prefix="${prefix:-$repository_root/.ovlab-env}"
        prefix="$("$bootstrap_python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$prefix")" || return 1
        target="$prefix"
    fi

    "$bootstrap_python" "$repository_root/deploy/scripts/setup_ovlab_env.py" "$@" || return $?
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        echo "To activate this shell, run: source ./install-ovlab-env.sh"
        if [[ "$manager" == conda ]]; then
            echo "Manual activation: conda activate \"$target\""
        else
            echo "Manual activation: source \"$prefix/bin/activate\""
        fi
        return 0
    fi

    if [[ "$manager" == conda ]]; then
        conda_hook="$(conda shell.bash hook)" || return 1
        eval "$conda_hook" || return 1
        conda activate "$target" || return 1
    else
        source "$prefix/bin/activate" || return 1
    fi
    echo "OVLAB environment active: $target"
    ovlab --version
}

_ovlab_env_setup "$@"
_ovlab_env_status=$?
unset -f _ovlab_env_setup
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exit "$_ovlab_env_status"
fi
return "$_ovlab_env_status"
