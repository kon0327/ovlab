"""Versioned prompt formatting matching pinned OpenVLA commit c8f03f48."""

from dataclasses import dataclass
from enum import Enum


class OpenVlaPromptTemplate(str, Enum):
    OPENVLA_V1 = "openvla-v1"
    OPENVLA_V01_CHAT = "openvla-v01-chat"


_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass(frozen=True, slots=True)
class OpenVlaPromptFormatter:
    template: OpenVlaPromptTemplate = OpenVlaPromptTemplate.OPENVLA_V1
    version: str = "1.0.0"

    @property
    def identifier(self) -> str:
        return f"{self.template.value}@{self.version}"

    def format(self, instruction: str) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must not be empty")
        # This lowercasing is required by the pinned OpenVLA inference utility.
        normalized = instruction.lower()
        if self.template is OpenVlaPromptTemplate.OPENVLA_V01_CHAT:
            return f"{_V01_SYSTEM_PROMPT} USER: What action should the robot take to {normalized}? ASSISTANT:"
        return f"In: What action should the robot take to {normalized}?\nOut:"
