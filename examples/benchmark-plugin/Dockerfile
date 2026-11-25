# syntax=docker/dockerfile:1.7
# Development fixture: build the shared reporting image first. No GPU is used.
FROM ovlab-reporting:local
USER root
COPY code /opt/ovlab/code
COPY deploy/docker/install-ovlab-packages.sh /tmp/install-ovlab-packages.sh
COPY external/extensions/example-benchmark/self /opt/ovlab/external/extensions/example-benchmark/self
ARG OVLAB_CONFIG_BUNDLE_SHA256=none
RUN --mount=type=secret,id=ovlab_plugin_registry \
    --mount=type=secret,id=ovlab_config_bundle \
    test -n "$OVLAB_CONFIG_BUNDLE_SHA256" \
    && bash /tmp/install-ovlab-packages.sh \
    && python -m pip install --no-deps --no-build-isolation /opt/ovlab/external/extensions/example-benchmark/self \
    && rm /tmp/install-ovlab-packages.sh \
    && install -d -o 10001 -g 10001 -m 0700 /run/ovlab

ARG OVLAB_REVISION
ARG OVLAB_SOURCE_MANIFEST_SHA256
ARG OVLAB_SOURCE_DIRTY
ARG OVLAB_LOCK_SHA256
ARG OVLAB_DOCKERFILE_SHA256
ARG OVLAB_IMAGE_REFERENCE
ARG OVLAB_VERSION
LABEL io.github.ovlab.deployment.contract="benchmark-rpc-v2-config-bundle-v5" \
      io.github.ovlab.role="benchmark-example" \
      org.opencontainers.image.revision="$OVLAB_REVISION" \
      org.opencontainers.image.version="$OVLAB_VERSION" \
      io.github.ovlab.source-manifest.sha256="$OVLAB_SOURCE_MANIFEST_SHA256" \
      io.github.ovlab.source-dirty="$OVLAB_SOURCE_DIRTY" \
      io.github.ovlab.dependency-lock.sha256="$OVLAB_LOCK_SHA256" \
      io.github.ovlab.dockerfile.sha256="$OVLAB_DOCKERFILE_SHA256"
ENV OVLAB_REVISION=$OVLAB_REVISION \
    OVLAB_IMAGE_ROLE=benchmark-example \
    OVLAB_IMAGE_REFERENCE=$OVLAB_IMAGE_REFERENCE \
    OVLAB_VERSION=$OVLAB_VERSION \
    OVLAB_SOURCE_MANIFEST_SHA256=$OVLAB_SOURCE_MANIFEST_SHA256 \
    OVLAB_SOURCE_DIRTY=$OVLAB_SOURCE_DIRTY \
    OVLAB_LOCK_SHA256=$OVLAB_LOCK_SHA256 \
    OVLAB_DOCKERFILE_SHA256=$OVLAB_DOCKERFILE_SHA256
RUN --mount=type=secret,id=ovlab_source_manifest \
    --mount=type=secret,id=ovlab_plugin_registry \
    --mount=type=secret,id=ovlab_config_bundle \
    test -s /run/secrets/ovlab_source_manifest \
    && cp /run/secrets/ovlab_source_manifest /usr/share/ovlab/source-manifest.json \
    && chmod 0444 /usr/share/ovlab/source-manifest.json
USER 10001:10001
ENTRYPOINT ["ovlab"]
CMD ["--help"]
