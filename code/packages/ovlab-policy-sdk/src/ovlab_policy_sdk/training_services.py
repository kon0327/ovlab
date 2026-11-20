"""Explicit host service contract for optional plugin checkpoint handoff hooks.

The SDK defines the dependency direction. OVLAB supplies these callbacks at its
extension boundary; plugins never import orchestration implementation modules.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol


class TrainingHostServices(Protocol):
    checkpoint_spec: object
    checkpoint_spec_from_snapshot: object
    default_global_cache: object
    resolve_finalized_training_checkpoint: object
    verify_checkpoint: object
    local_checkpoint_override: object
    training_store: object
    checkpoint_store: object
    resolve_run_reference: object
    validate_document: object
    image_environment: object


_SERVICES = ContextVar("ovlab_training_host_services", default=None)


def host_services() -> TrainingHostServices:
    services = _SERVICES.get()
    if services is None:
        raise RuntimeError("plugin checkpoint handoff requires OVLAB host services")
    return services


@contextmanager
def training_host_scope(services: TrainingHostServices):
    token = _SERVICES.set(services)
    try:
        yield
    finally:
        _SERVICES.reset(token)
