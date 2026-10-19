"""Model-independent QuIC primitives; no benchmark or host-model imports."""

from .adapter import (
    AdapterConfig, CompoundOperator, OrthogonalBase, QuICLinear,
    adapter_manifest, adapter_parameters, compound_matrix, disabled_adapters,
    inject, load_adapter, merge_adapters, save_adapter,
)

__all__ = [
    "AdapterConfig", "CompoundOperator", "OrthogonalBase", "QuICLinear",
    "adapter_manifest", "adapter_parameters", "compound_matrix", "disabled_adapters",
    "inject", "load_adapter", "merge_adapters", "save_adapter",
]
