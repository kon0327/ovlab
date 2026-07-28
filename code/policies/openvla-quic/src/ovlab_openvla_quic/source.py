"""Dependency-free identity and configuration mapping for the legacy source intake."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

from .errors import QuICDescriptorError

COMPOUND_PEFT_ARCHIVE_SHA256 = "b024ba61b852d83beec631b724489b3bc3055c4a883f2df0c05b6c9857103e9a"
COMPOUND_PEFT_MANIFEST_SHA256 = "8084213849149a47f9bf84dd0c9220b319faf7df8dba39cdef3894e85e00f845"
COMPOUND_PEFT_FILE_COUNT = 130
COMPOUND_PEFT_PACKAGE_VERSION = "0.12.1.dev0"
COMPOUND_PEFT_PATH = "external/compound-peft"
SUPPORTED_COMPOUND_PATTERNS = ("comp_1", "comp_2", "comp_3")
SUPPORTED_COMPOUND_TYPES = ("comp", "max", "avg", "perm")

PAPER_IDENTITY = {
    "title": "QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning",
    "authors": ["Snehal Raj", "Brian Coyle"],
    "arxiv": "2502.06916",
}

SOURCE_LIMITATIONS = {
    "paper_pretrained_matrix_shape": "square_assumed",
    "implementation_wrapped_module_types": ["torch.nn.Linear", "torch.nn.Conv2d"],
    "rectangular_openvla_projection_applicability": "unverified",
    "target_module_placement": "unresolved",
    "dense_adapter_materialization": True,
    "determinant_compute_dtype": "float32_promoted",
    "merge_unmerge_validation": "implemented_unvalidated",
    "checkpoint_round_trip_validation": "implemented_unvalidated",
    "paper_forward_merge_numerical_equivalence": "unvalidated",
    "complete_base_model_required": True,
    "weight_compression": False,
}


def compound_peft_source_identity() -> dict[str, object]:
    return {
        "availability": "available",
        "source_path": COMPOUND_PEFT_PATH,
        "source_origin": "user_supplied_archive",
        "source_revision_kind": "content_hash",
        "archive_sha256": COMPOUND_PEFT_ARCHIVE_SHA256,
        "archive_verification": "user_supplied_archive_not_locally_available",
        "archive_byte_size": "unavailable",
        "extracted_manifest_sha256": COMPOUND_PEFT_MANIFEST_SHA256,
        "file_count": COMPOUND_PEFT_FILE_COUNT,
        "upstream_git_revision": "unavailable",
        "official_implementation_status": "unverified",
        "relation_to_paper": "claimed_by_readme_unverified",
        "scientific_oracle_status": False,
        "package_name": "peft",
        "package_version": COMPOUND_PEFT_PACKAGE_VERSION,
        "license": "Apache-2.0",
        "paper_identity": PAPER_IDENTITY,
        "readme_citation_agrees_with_paper": False,
        "limitations": SOURCE_LIMITATIONS,
    }


@dataclass(frozen=True, slots=True)
class CompoundLegacyConfig:
    """Raw bundled fields; ``r`` is a block count and never a LoRA rank."""

    r: int
    compound_pattern: tuple[str, ...]
    compound_type: str
    block_share: bool
    use_orthogonal: bool
    num_adapters: int
    adapter_multiplicative: bool
    use_scaling: bool
    use_offset_blocks: bool

    def __post_init__(self) -> None:
        if type(self.r) is not int or self.r <= 0:
            raise QuICDescriptorError("legacy r must be a positive number of block-diagonal blocks")
        patterns = tuple(self.compound_pattern)
        if not patterns or len(patterns) != len(set(patterns)):
            raise QuICDescriptorError("compound_pattern must contain unique supported orders")
        unsupported = sorted(set(patterns) - set(SUPPORTED_COMPOUND_PATTERNS))
        if unsupported:
            raise QuICDescriptorError(
                f"legacy backend supports only comp_1, comp_2, and comp_3; got {unsupported}"
            )
        if self.compound_type not in SUPPORTED_COMPOUND_TYPES:
            raise QuICDescriptorError(
                f"legacy compound_type must be one of {SUPPORTED_COMPOUND_TYPES}"
            )
        for name in (
            "block_share", "use_orthogonal", "adapter_multiplicative",
            "use_scaling", "use_offset_blocks",
        ):
            if type(getattr(self, name)) is not bool:
                raise QuICDescriptorError(f"legacy {name} must be boolean")
        if type(self.num_adapters) is not int or self.num_adapters <= 0:
            raise QuICDescriptorError("legacy num_adapters must be positive")
        if self.use_offset_blocks and self.num_adapters == 1:
            raise QuICDescriptorError("offset blocks require multiple legacy adapters")
        object.__setattr__(self, "compound_pattern", patterns)

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "CompoundLegacyConfig":
        expected = {
            "r", "compound_pattern", "compound_type", "block_share", "use_orthogonal",
            "num_adapters", "adapter_multiplicative", "use_scaling", "use_offset_blocks",
        }
        if set(value) != expected:
            raise QuICDescriptorError(f"legacy compound fields must equal {sorted(expected)}")
        return cls(
            r=value["r"],
            compound_pattern=tuple(value["compound_pattern"]),
            compound_type=value["compound_type"],
            block_share=value["block_share"],
            use_orthogonal=value["use_orthogonal"],
            num_adapters=value["num_adapters"],
            adapter_multiplicative=value["adapter_multiplicative"],
            use_scaling=value["use_scaling"],
            use_offset_blocks=value["use_offset_blocks"],
        )

    def normalize(self, *, target_output_dimension: int | None = None) -> dict[str, object]:
        block_dimension = None
        if target_output_dimension is not None:
            if type(target_output_dimension) is not int or target_output_dimension <= 0:
                raise QuICDescriptorError("target output dimension must be positive or unavailable")
            if target_output_dimension % self.r:
                raise QuICDescriptorError(
                    "target output dimension must be divisible by the legacy block count"
                )
            block_dimension = target_output_dimension // self.r
        extensions = []
        if self.compound_type in {"max", "avg", "perm"}:
            extensions.append(f"compound_operation:{self.compound_type}")
        if self.num_adapters != 1:
            extensions.append("multi_adapter_chain")
        if not self.adapter_multiplicative:
            extensions.append("additive_adapter_composition")
        if self.use_scaling:
            extensions.append("learnable_scaling")
        if self.use_offset_blocks:
            extensions.append("offset_blocks")
        legacy = {
            "r": self.r,
            "compound_pattern": list(self.compound_pattern),
            "compound_type": self.compound_type,
            "block_share": self.block_share,
            "use_orthogonal": self.use_orthogonal,
            "num_adapters": self.num_adapters,
            "adapter_multiplicative": self.adapter_multiplicative,
            "use_scaling": self.use_scaling,
            "use_offset_blocks": self.use_offset_blocks,
        }
        operation = {
            "comp": "determinant",
            "max": "maximum",
            "avg": "average",
            "perm": "permanent",
        }[self.compound_type]
        return {
            "availability": "available",
            "legacy": legacy,
            "canonical": {
                "num_blocks": self.r,
                "block_dimension": block_dimension,
                "block_dimension_derivation": "target_output_dimension_divided_by_num_blocks",
                "compound_orders": [int(item.removeprefix("comp_")) for item in self.compound_pattern],
                "compound_operation": operation,
                "parameter_sharing": "shared_across_blocks" if self.block_share else "unshared_blocks",
                "orthogonality_enforcement": "cayley" if self.use_orthogonal else "none",
                "adapter_chain_length": self.num_adapters,
                "adapter_composition": "multiplicative" if self.adapter_multiplicative else "additive",
                "learnable_scaling": self.use_scaling,
                "offset_blocks": self.use_offset_blocks,
            },
            "semantic_guards": {
                "r_semantics": "number_of_block_diagonal_blocks_not_lora_rank",
                "r_to_paper_b_translation": "unresolved_not_directly_equated",
                "arbitrary_compound_order_supported": False,
                "paper_equivalence_validated": False,
            },
            "field_classification": {
                "paper_formulation_fields": [
                    "canonical.compound_orders", "canonical.compound_operation:determinant",
                    "canonical.orthogonality_enforcement",
                ],
                "implementation_extensions_active": extensions,
                "implementation_extensions_known": [
                    "compound_operation:max", "compound_operation:avg", "compound_operation:perm",
                    "multi_adapter_chain", "additive_adapter_composition", "learnable_scaling",
                    "offset_blocks",
                ],
            },
        }


def verify_compound_peft_tree(root: Path) -> dict[str, object]:
    """Reproduce the immutable manifest without importing any supplied code."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise QuICDescriptorError(f"compound-peft source tree is absent: {root}")
    files = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise QuICDescriptorError(f"compound-peft tree contains a directory link: {directory_path}")
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink():
                raise QuICDescriptorError(f"compound-peft tree contains an unexpected link: {path}")
            if path.suffix == ".pyc":
                continue
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append((relative, digest))
    files.sort()
    manifest = "".join(f"{digest}  ./{relative}\n" for relative, digest in files).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    if manifest_digest != COMPOUND_PEFT_MANIFEST_SHA256:
        raise QuICDescriptorError(
            f"unexplained compound-peft divergence: manifest={manifest_digest}"
        )
    if len(files) != COMPOUND_PEFT_FILE_COUNT:
        raise QuICDescriptorError("compound-peft file count differs from the recorded identity")
    license_text = (root / "peft/LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0, January 2004" not in license_text:
        raise QuICDescriptorError("compound-peft Apache-2.0 license identity is missing")
    init_text = (root / "peft/src/peft/__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    if version_match is None or version_match.group(1) != COMPOUND_PEFT_PACKAGE_VERSION:
        raise QuICDescriptorError("compound-peft package version differs from the recorded identity")
    return {
        "source_path": str(root),
        "manifest_sha256": manifest_digest,
        "file_count": len(files),
        "license": "Apache-2.0",
        "package_version": version_match.group(1),
        "divergence_classification": "none_manifest_matches",
    }


def statically_compile_compound_peft(root: Path) -> int:
    """Compile source text in memory; no imports and no generated bytecode."""

    count = 0
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        compile(path.read_bytes(), str(path), "exec")
        count += 1
    return count
