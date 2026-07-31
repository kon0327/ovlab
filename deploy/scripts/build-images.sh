#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
python deploy/scripts/source_manifest.py --root "$repository_root" --output "$manifest"

revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository_revision"])' "$manifest")"
source_sha="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_content_sha256"])' "$manifest")"
dirty="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["dirty"]).lower())' "$manifest")"
version="$(tr -d '[:space:]' < VERSION)"

build_role() {
  local role="$1"
  local dockerfile="$2"
  local image="$3"
  shift 3
  local lock_sha dockerfile_sha
  lock_sha="$(sha256sum deploy/docker/base-images.lock "$@" | sha256sum | cut -d' ' -f1)"
  dockerfile_sha="$(sha256sum "$dockerfile" | cut -d' ' -f1)"
  docker build --pull=false --file "$dockerfile" --tag "$image" \
    --secret "id=ovlab_source_manifest,src=$manifest" \
    --build-arg "OVLAB_REVISION=$revision" \
    --build-arg "OVLAB_SOURCE_MANIFEST_SHA256=$source_sha" \
    --build-arg "OVLAB_SOURCE_DIRTY=$dirty" \
    --build-arg "OVLAB_LOCK_SHA256=$lock_sha" \
    --build-arg "OVLAB_DOCKERFILE_SHA256=$dockerfile_sha" \
    --build-arg "OVLAB_IMAGE_REFERENCE=$image" \
    --build-arg "OVLAB_VERSION=$version" \
    .
  printf '%s image=%s id=%s source=%s dirty=%s lock=%s dockerfile=%s\n' \
    "$role" "$image" "$(docker image inspect --format '{{.Id}}' "$image")" \
    "$source_sha" "$dirty" "$lock_sha" "$dockerfile_sha"
}

selected() {
  local candidate="$1"
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  local requested
  for requested in "${roles[@]}"; do
    [[ "$requested" == "$candidate" ]] && return 0
  done
  return 1
}

roles=("$@")
for requested in "${roles[@]}"; do
  case "$requested" in
    benchmark|reporting|dataset|training-openvla|policy-openvla|policy-openvla-oft) ;;
    *) printf 'unknown image role: %s\n' "$requested" >&2; exit 2 ;;
  esac
done

if [[ "${#roles[@]}" -eq 0 ]] || selected benchmark; then
  build_role benchmark deploy/docker/Dockerfile.benchmark ovlab-benchmark-libero:local \
    deploy/locks/benchmark.pylock.toml deploy/locks/benchmark.requirements.txt
fi
if [[ "${#roles[@]}" -eq 0 ]] || selected reporting; then
  build_role reporting deploy/docker/Dockerfile.reporting ovlab-reporting:local \
    deploy/locks/reporting.pylock.toml deploy/locks/reporting.requirements.txt
fi
if [[ "${#roles[@]}" -eq 0 ]] || selected dataset; then
  build_role dataset deploy/docker/Dockerfile.dataset ovlab-dataset:local \
    deploy/locks/dataset.pylock.toml deploy/locks/dataset.requirements.txt
fi
if [[ "${#roles[@]}" -eq 0 ]] || selected training-openvla; then
  build_role training-openvla deploy/docker/Dockerfile.training-openvla ovlab-training-openvla:local \
    deploy/locks/openvla-oft.pylock.toml deploy/locks/openvla-oft.requirements.txt \
    deploy/locks/training-openvla.pylock.toml deploy/locks/training-openvla.requirements.txt \
    deploy/locks/flash-attn.pylock.toml deploy/locks/flash-attn.requirements.txt
fi
if [[ "${#roles[@]}" -eq 0 ]] || selected policy-openvla; then
  build_role policy-openvla deploy/docker/Dockerfile.openvla ovlab-policy-openvla:local \
    deploy/locks/openvla.pylock.toml deploy/locks/openvla.requirements.txt \
    deploy/locks/flash-attn.pylock.toml deploy/locks/flash-attn.requirements.txt
fi
if [[ "${#roles[@]}" -eq 0 ]] || selected policy-openvla-oft; then
  build_role policy-openvla-oft deploy/docker/Dockerfile.openvla-oft ovlab-policy-openvla-oft:local \
    deploy/locks/openvla-oft.pylock.toml deploy/locks/openvla-oft.requirements.txt \
    deploy/locks/flash-attn.pylock.toml deploy/locks/flash-attn.requirements.txt
fi
