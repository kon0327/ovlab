#!/usr/bin/env bash
set -euo pipefail
packages=()
for project in /opt/ovlab/code/packages/* /opt/ovlab/code/apps/*; do
  if [[ -f "$project/pyproject.toml" ]]; then packages+=("$project"); fi
done
python -m pip install --no-cache-dir --no-deps --no-build-isolation "${packages[@]}"

# A build supplies the resolved registry independently of the core packages.
# Infrastructure images can describe installed services without importing models.
if [[ -s /run/secrets/ovlab_plugin_registry ]]; then
  install -d -o root -g root -m 0755 /usr/share/ovlab
  install -o root -g root -m 0444 /run/secrets/ovlab_plugin_registry /usr/share/ovlab/plugin-registry.json
fi

# Optional explicit experiment/deployment closure; all archive members originate
# from ConfigBundleBuilder's bounded, regular configuration files.
if [[ -s /run/secrets/ovlab_config_bundle ]]; then
  : "${OVLAB_CONFIG_BUNDLE_SHA256:?configured builds require a cache-bound bundle digest}"
  install -d -o root -g root -m 0755 /opt/ovlab/configs
  tar --extract --file /run/secrets/ovlab_config_bundle --directory /opt/ovlab/configs --no-same-owner
  python - <<'PY'
import json
import os
from pathlib import Path
manifest = json.loads(Path('/opt/ovlab/configs/.ovlab-bundle.json').read_text())
if manifest['bundle_sha256'] != os.environ['OVLAB_CONFIG_BUNDLE_SHA256']:
    raise SystemExit('configuration bundle differs from its build cache identity')
PY
  chmod -R a+rX /opt/ovlab/configs
fi
