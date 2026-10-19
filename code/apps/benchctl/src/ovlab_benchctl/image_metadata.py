"""Canonical OCI metadata used by OVLAB deployment images."""

from __future__ import annotations


IMAGE_LABEL_NAMESPACE = "io.github.ovlab"
IMAGE_ROLE_LABEL = f"{IMAGE_LABEL_NAMESPACE}.role"
IMAGE_CONTRACT_LABEL = f"{IMAGE_LABEL_NAMESPACE}.deployment.contract"
SOURCE_MANIFEST_LABEL = f"{IMAGE_LABEL_NAMESPACE}.source-manifest.sha256"

