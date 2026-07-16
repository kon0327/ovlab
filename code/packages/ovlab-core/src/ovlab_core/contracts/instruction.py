"""Timestamped static and dynamically replaced instructions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Mapping

from .errors import validation_error
from .identifiers import InstructionId
from .metadata import Metadata, normalize_metadata
from .time import validate_timestamp_ns


class InstructionSource(str, Enum):
    BENCHMARK = "benchmark"
    USER = "user"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Instruction:
    instruction_id: InstructionId
    text: str
    timestamp_ns: int
    source: InstructionSource
    supersedes: InstructionId | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        if not isinstance(self.instruction_id, InstructionId):
            raise validation_error(contract, "instruction_id", "must be an InstructionId")
        if not isinstance(self.text, str) or not self.text.strip():
            raise validation_error(contract, "text", "must not be empty or whitespace-only")
        validate_timestamp_ns(self.timestamp_ns, contract, "timestamp_ns")
        if not isinstance(self.source, InstructionSource):
            raise validation_error(contract, "source", "must be an InstructionSource")
        if self.supersedes is not None and not isinstance(self.supersedes, InstructionId):
            raise validation_error(contract, "supersedes", "must be an InstructionId or None")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))
