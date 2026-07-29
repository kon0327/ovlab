#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

docker build --file deploy/docker/Dockerfile.smoke --target smoke \
  --tag ovlab-transport-smoke:local .
docker compose --file deploy/compose/compose.smoke.yaml up \
  --abort-on-container-exit --exit-code-from benchmark
docker compose --file deploy/compose/compose.smoke.yaml down --volumes --remove-orphans
