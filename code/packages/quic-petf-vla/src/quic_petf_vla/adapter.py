"""Differentiable, output-side QuIC adapters over frozen dense linear layers.

The compound coordinates use ascending, lexicographic combinations. Each block
contains the requested compound orders followed by unchanged coordinates. Bias
is added after rotating the weight projection, as specified in QuIC.md.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_CONSTRUCTION_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AdapterConfig:
    orders: tuple[int, ...] = (1, 2)
    block_count: int = 8
    base_size: int | None = None
    share_across_blocks: bool = False
    orthogonal: bool = True
    orientation: str = "output"
    construction_dtype: str = "float32"

    def __post_init__(self) -> None:
        orders = tuple(self.orders)
        if not orders or any(type(k) is not int or k not in (1, 2, 3) for k in orders):
            raise ValueError("orders must be a nonempty selection of integer orders 1, 2, 3")
        if tuple(sorted(set(orders))) != orders:
            raise ValueError("orders must be unique and in ascending order")
        object.__setattr__(self, "orders", orders)
        if type(self.block_count) is not int or self.block_count < 1:
            raise ValueError("block_count must be a positive integer")
        if self.base_size is not None and (type(self.base_size) is not int or self.base_size < max(2, max(orders))):
            raise ValueError("base_size must be at least 2 and at least the largest order")
        if type(self.share_across_blocks) is not bool or type(self.orthogonal) is not bool:
            raise ValueError("share_across_blocks and orthogonal must be boolean")
        if self.orientation != "output":
            raise ValueError("only orientation='output' is implemented")
        if self.construction_dtype not in _CONSTRUCTION_DTYPES:
            raise ValueError("construction_dtype must be float32 or float64")

    def resolve(self, out_features: int) -> AdapterConfig:
        if type(out_features) is not int or out_features < 1:
            raise ValueError("out_features must be a positive integer")
        if out_features % self.block_count:
            raise ValueError(f"out_features={out_features} is not divisible by block_count={self.block_count}")
        size = out_features // self.block_count
        n = self.base_size
        minimum = max(2, max(self.orders))
        if n is None:
            if sum(math.comb(minimum, k) for k in self.orders) > size:
                raise ValueError(f"compound orders {self.orders} do not fit block size {size}")
            # Binary search keeps C1-only configurations inexpensive to resolve.
            lo, hi = minimum, size
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if sum(math.comb(mid, k) for k in self.orders) <= size:
                    lo = mid
                else:
                    hi = mid - 1
            n = lo
        if sum(math.comb(n, k) for k in self.orders) > size:
            raise ValueError(f"base_size={n}, orders={self.orders} exceed block size {size}")
        return replace(self, base_size=n)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["orders"] = list(self.orders)
        return result


def _combinations(n: int, order: int, device: torch.device) -> Tensor:
    return torch.tensor(list(itertools.combinations(range(n), order)), dtype=torch.long, device=device)


def compound_matrix(base: Tensor, order: int, indices: Tensor | None = None) -> Tensor:
    """Return C_k(base), including batch dimensions, using explicit polynomials.

    Explicit 2x2 and 3x3 formulas retain correct derivatives at singular minors
    occurring at identity initialization. No inverse or general determinant is
    used in the compound calculation.
    """
    if base.ndim < 2 or base.shape[-1] != base.shape[-2]:
        raise ValueError("base must contain square matrices")
    if type(order) is not int or order not in (1, 2, 3) or order > base.shape[-1]:
        raise ValueError("compound order must be 1, 2, or 3 and no larger than the base")
    if order == 1:
        return base
    idx = _combinations(base.shape[-1], order, base.device) if indices is None else indices
    # Each component is a batched matrix indexed by pairs of combinations.
    def entry(row: int, col: int) -> Tensor:
        return base[..., idx[:, row, None], idx[None, :, col]]

    a, b, d, e = entry(0, 0), entry(0, 1), entry(1, 0), entry(1, 1)
    if order == 2:
        return a * e - b * d
    c, f = entry(0, 2), entry(1, 2)
    g, h, i = entry(2, 0), entry(2, 1), entry(2, 2)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class OrthogonalBase(nn.Module):
    """Minimal skew parameters and a differentiable Cayley linear solve."""

    def __init__(self, size: int, count: int, *, device: torch.device, dtype: torch.dtype,
                 orthogonal: bool = True) -> None:
        super().__init__()
        self.size = size
        self.count = count
        self.orthogonal = orthogonal
        self.construction_dtype = dtype
        if orthogonal:
            self.skew_entries = nn.Parameter(torch.zeros(count, size * (size - 1) // 2, device=device, dtype=dtype))
        else:
            self.matrix_delta = nn.Parameter(torch.zeros(count, size, size, device=device, dtype=dtype))
        self.register_buffer("upper_indices", torch.triu_indices(size, size, 1, device=device), persistent=False)

    @property
    def parameter(self) -> nn.Parameter:
        return self.skew_entries if self.orthogonal else self.matrix_delta

    def _apply(self, fn: Any, recurse: bool = True) -> OrthogonalBase:
        super()._apply(fn, recurse=recurse)
        # A host .to(bfloat16/float16) keeps learned adapter entries in the
        # declared construction precision while moving them to the new device.
        self.parameter.data = self.parameter.data.to(dtype=self.construction_dtype)
        if self.parameter.grad is not None:
            self.parameter.grad.data = self.parameter.grad.data.to(dtype=self.construction_dtype)
        return self

    def forward(self, construction_dtype: torch.dtype) -> Tensor:
        entries = self.parameter.to(dtype=construction_dtype)
        identity = torch.eye(self.size, device=entries.device, dtype=entries.dtype).expand(self.count, -1, -1)
        if not self.orthogonal:
            return identity + entries
        skew = entries.new_zeros(self.count, self.size, self.size)
        rows, cols = self.upper_indices
        skew[:, rows, cols] = entries
        skew[:, cols, rows] = -entries
        return torch.linalg.solve((identity - skew).transpose(-1, -2), (identity + skew).transpose(-1, -2)).transpose(-1, -2)


class CompoundOperator(nn.Module):
    """Apply compound matrices within each block without a full output matrix."""

    def __init__(self, out_features: int, config: AdapterConfig, *, device: torch.device) -> None:
        super().__init__()
        self.config = config.resolve(out_features)
        self.out_features = out_features
        self.block_size = out_features // self.config.block_count
        self.compound_sizes = tuple(math.comb(self.config.base_size, k) for k in self.config.orders)
        self.compound_dimension = sum(self.compound_sizes)
        self.padding_per_block = self.block_size - self.compound_dimension
        self.base = OrthogonalBase(
            self.config.base_size, 1 if self.config.share_across_blocks else self.config.block_count,
            device=device, dtype=_CONSTRUCTION_DTYPES[self.config.construction_dtype],
            orthogonal=self.config.orthogonal,
        )
        for order in self.config.orders:
            if order > 1:
                self.register_buffer(f"indices_{order}", _combinations(self.config.base_size, order, device), persistent=False)

    @property
    def skew_entries(self) -> nn.Parameter:
        return self.base.skew_entries

    @property
    def parameter(self) -> nn.Parameter:
        return self.base.parameter

    def matrices(self) -> tuple[Tensor, ...]:
        # Host AMP must not lower the Cayley solve or polynomial construction.
        with torch.autocast(device_type=self.parameter.device.type, enabled=False):
            base = self.base(_CONSTRUCTION_DTYPES[self.config.construction_dtype])
            return tuple(compound_matrix(base, order, getattr(self, f"indices_{order}", None)) for order in self.config.orders)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.out_features:
            raise ValueError(f"expected {self.out_features} output coordinates")
        blocks = values.reshape(*values.shape[:-1], self.config.block_count, self.block_size)
        pieces = []
        start = 0
        for size, matrix in zip(self.compound_sizes, self.matrices()):
            matrix = matrix.to(dtype=values.dtype)
            if self.config.share_across_blocks:
                matrix = matrix.expand(self.config.block_count, -1, -1)
            part = blocks[..., start:start + size]
            pieces.append(torch.matmul(part.unsqueeze(-2), matrix.transpose(-1, -2)).squeeze(-2))
            start += size
        if self.padding_per_block:
            pieces.append(blocks[..., start:])
        return torch.cat(pieces, dim=-1).reshape_as(values)


class QuICLinear(nn.Module):
    """Output rotation with unchanged bias and exact identity initialization.

    Bias-bearing layers use host(x) + (R z - z), z = W x. This is algebraically
    R W x + b while preserving fused host bias rounding at identity, including
    BF16. The reference implementation pays for a second host projection in
    bias-bearing layers; bias-free backbone projections are computed once.
    """

    def __init__(self, base_layer: nn.Linear, config: AdapterConfig) -> None:
        super().__init__()
        _validate_linear(base_layer)
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.adapter = CompoundOperator(self.out_features, config, device=base_layer.weight.device)
        self.config = self.adapter.config
        self.adapter_enabled = True

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, values: Tensor) -> Tensor:
        if not self.adapter_enabled:
            return F.linear(values, self.base_layer.weight, self.base_layer.bias)
        projected = F.linear(values, self.base_layer.weight, None)
        rotated = self.adapter(projected)
        if self.base_layer.bias is not None:
            rotated = self.base_layer(values) + (rotated - projected)
        return rotated

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        weight = self.adapter(self.weight.transpose(0, 1)).transpose(0, 1).contiguous()
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device=self.weight.device, dtype=self.weight.dtype)
        merged.weight.copy_(weight)
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        merged.requires_grad_(False)
        merged.train(self.training)
        return merged


def _validate_linear(layer: nn.Module) -> None:
    if type(layer) is not nn.Linear:
        raise TypeError("QuIC supports plain torch.nn.Linear only; quantized or specialized subclasses are unsupported")
    if not layer.weight.is_floating_point() or layer.weight.device.type == "meta":
        raise ValueError("host weights must be materialized floating-point tensors")


def _describe(name: str, layer: nn.Linear | QuICLinear, config: AdapterConfig) -> dict[str, Any]:
    resolved = config.resolve(layer.out_features)
    dimension = sum(math.comb(resolved.base_size, k) for k in resolved.orders)
    block_size = layer.out_features // resolved.block_count
    count = 1 if resolved.share_across_blocks else resolved.block_count
    parameters = count * (resolved.base_size * (resolved.base_size - 1) // 2 if resolved.orthogonal else resolved.base_size ** 2)
    return {
        "name": name, "in_features": layer.in_features, "out_features": layer.out_features,
        "bias": layer.bias is not None, "config": resolved.to_dict(),
        "block_size": block_size, "compound_dimension_per_block": dimension,
        "identity_padding_per_block": block_size - dimension,
        "adapted_coordinate_fraction": dimension / block_size,
        "independent_parameters": parameters, "allocated_parameters": parameters,
        "parameterization": "minimal_skew_cayley" if resolved.orthogonal else "identity_plus_full_delta",
        "index_order": "ascending_lexicographic", "bias_policy": "unchanged",
    }


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def inject(model: nn.Module, targets: Mapping[str, AdapterConfig]) -> list[dict[str, Any]]:
    """Validate every explicit target before freezing and modifying the host."""
    if not targets:
        raise ValueError("at least one explicit target is required")
    prepared = []
    for name, config in targets.items():
        if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError("target names must be nonempty module paths")
        if not isinstance(config, AdapterConfig):
            raise TypeError("target values must be AdapterConfig instances")
        try:
            layer = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"target module {name!r} does not exist") from error
        _validate_linear(layer)
        prepared.append((name, layer, config.resolve(layer.out_features), _describe(name, layer, config)))
    if len({id(layer) for _, layer, _, _ in prepared}) != len(prepared):
        raise ValueError("targets alias the same Linear; weight-sharing requires an explicit strategy")
    if any(isinstance(module, QuICLinear) for module in model.modules()):
        raise ValueError("host already contains QuIC adapters")
    model.requires_grad_(False)
    for name, layer, config, _ in prepared:
        wrapper = QuICLinear(layer, config)
        wrapper.train(layer.training)
        _replace_module(model, name, wrapper)
    return [description for _, _, _, description in prepared]


def adapter_manifest(model: nn.Module) -> list[dict[str, Any]]:
    return [_describe(name, module, module.config) for name, module in model.named_modules() if isinstance(module, QuICLinear)]


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [module.adapter.parameter for module in model.modules() if isinstance(module, QuICLinear)]


@contextmanager
def disabled_adapters(model: nn.Module):
    """Temporarily evaluate the frozen original host, including nested contexts."""
    previous = [(module, module.adapter_enabled) for module in model.modules() if isinstance(module, QuICLinear)]
    try:
        for module, _ in previous:
            module.adapter_enabled = False
        yield model
    finally:
        for module, enabled in previous:
            module.adapter_enabled = enabled


def merge_adapters(model: nn.Module, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Replace selected adapters with dense RW host weights, keeping bias fixed."""
    wrappers = {name: module for name, module in model.named_modules() if isinstance(module, QuICLinear)}
    selected = list(wrappers) if names is None else list(names)
    if len(set(selected)) != len(selected) or any(name not in wrappers for name in selected):
        raise ValueError("merge names must be distinct existing QuIC layers")
    manifest = [_describe(name, wrappers[name], wrappers[name].config) for name in selected]
    for name in selected:
        _replace_module(model, name, wrappers[name].merged_linear())
    return manifest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_host_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("host_identity must be a nonempty JSON object with an immutable host hash")
    result = json.loads(_canonical(dict(identity)))
    # Accommodate callers naming their checkpoint/tree digest explicitly.
    digest_values = [v for k, v in result.items() if "sha256" in k or k in ("checkpoint_hash", "host_hash")]
    if not any(isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v) for v in digest_values):
        raise ValueError("host_identity must include a 64-character SHA256 digest")
    return result


def _tensor_key(name: str) -> str:
    return f"{name}.adapter.entries"


def save_adapter(model: nn.Module, directory: str | Path, host_identity: Mapping[str, Any],
                 metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write only minimal learned skew entries plus a strict host/target manifest."""
    from safetensors.torch import save_file

    identity = _validate_host_identity(host_identity)
    manifest = adapter_manifest(model)
    if not manifest:
        raise ValueError("model contains no QuIC adapters")
    params = adapter_parameters(model)
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in params}:
        raise ValueError("only adapter parameters may be trainable when saving")
    if any(not torch.isfinite(p).all() for p in params):
        raise ValueError("adapter parameters contain nonfinite values")
    _canonical(dict(metadata or {}))
    binding = {"format_version": _FORMAT_VERSION, "host_identity": identity, "manifest": manifest}
    binding_sha = hashlib.sha256(_canonical(binding)).hexdigest()
    tensors = {_tensor_key(name): module.adapter.parameter.detach().cpu().contiguous()
               for name, module in model.named_modules() if isinstance(module, QuICLinear)}
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "adapter.json").exists() or (target / "adapter.safetensors").exists():
        raise FileExistsError("adapter artifact already exists; use a new directory to preserve provenance")
    tensor_path = target / "adapter.safetensors"
    save_file(tensors, str(tensor_path), metadata={"quic_binding_sha256": binding_sha})
    payload = {**binding, "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
               "metadata": dict(metadata or {}), "torch_version": torch.__version__}
    (target / "adapter.json").write_bytes(_canonical(payload) + b"\n")
    return payload


def load_adapter(model: nn.Module, directory: str | Path, host_identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify identity, manifest, tensor digest and shapes before injecting."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    identity = _validate_host_identity(host_identity)
    directory = Path(directory)
    payload = json.loads((directory / "adapter.json").read_text(encoding="utf-8"))
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported QuIC artifact version")
    if payload.get("host_identity") != identity:
        raise ValueError("adapter host identity does not match the requested host")
    tensor_path = directory / "adapter.safetensors"
    if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != payload.get("tensor_sha256"):
        raise ValueError("adapter tensor SHA256 mismatch")
    binding = {key: payload[key] for key in ("format_version", "host_identity", "manifest")}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as saved:
        if (saved.metadata() or {}).get("quic_binding_sha256") != hashlib.sha256(_canonical(binding)).hexdigest():
            raise ValueError("adapter manifest binding mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    targets = {}
    for entry in payload["manifest"]:
        name = entry["name"]
        if name in targets:
            raise ValueError("duplicate target in adapter artifact")
        config = AdapterConfig(**entry["config"])
        try:
            layer = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"saved target {name!r} is missing from host") from error
        _validate_linear(layer)
        if _describe(name, layer, config) != entry:
            raise ValueError(f"saved target manifest does not match host layer {name!r}")
        key = _tensor_key(name)
        n = config.base_size
        count = 1 if config.share_across_blocks else config.block_count
        shape = (count, n * (n - 1) // 2) if config.orthogonal else (count, n, n)
        if key not in tensors or tuple(tensors[key].shape) != shape:
            raise ValueError(f"saved parameter shape is invalid for {name!r}")
        if tensors[key].dtype != _CONSTRUCTION_DTYPES[config.construction_dtype] or not torch.isfinite(tensors[key]).all():
            raise ValueError(f"saved parameter dtype or values are invalid for {name!r}")
        targets[name] = config
    if set(tensors) != {_tensor_key(name) for name in targets}:
        raise ValueError("adapter artifact contains unexpected parameters")
    manifest = inject(model, targets)
    with torch.no_grad():
        for name in targets:
            model.get_submodule(name).adapter.parameter.copy_(tensors[_tensor_key(name)])
    return manifest
