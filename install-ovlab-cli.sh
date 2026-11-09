#!/usr/bin/env bash
# Install a user-local launcher that delegates to this checkout's OVLAB script.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
bin_dir="${OVLAB_BIN_DIR:-$HOME/.local/bin}"
mkdir -p -- "$bin_dir"
bin_dir="$(cd "$bin_dir" && pwd -P)"
target="$bin_dir/ovlab"
temporary="$(mktemp "$bin_dir/.ovlab-launcher.XXXXXXXX")"
trap 'rm -f -- "$temporary"' EXIT

printf '#!/usr/bin/env bash\n# OVLAB checkout launcher\nexec %q "$@"\n' "$repository_root/ovlab" > "$temporary"
chmod 0755 -- "$temporary"
if [[ -e "$target" || -L "$target" ]]; then
    if [[ ! -L "$target" && -f "$target" ]] && cmp -s -- "$temporary" "$target"; then
        echo "OVLAB launcher already installed: $target"
    else
        echo "ovlab: refusing to replace existing command: $target" >&2
        exit 1
    fi
else
    mv -n -- "$temporary" "$target"
    if [[ -e "$temporary" ]]; then
        echo "ovlab: command appeared during installation: $target" >&2
        exit 1
    fi
    echo "OVLAB launcher installed: $target"
fi

case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) printf 'Add this directory to PATH: export PATH=%q:"$PATH"\n' "$bin_dir" ;;
esac
