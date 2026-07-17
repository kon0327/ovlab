"""Immutable portable and resolved configuration evidence for one run."""

from dataclasses import dataclass


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _sha256(value: str, field_name: str) -> None:
    _required_text(value, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RunConfigurationSnapshot:
    """Serialized config evidence; artifact stores persist it without interpreting YAML."""

    portable_source_yaml: str
    resolved_config_yaml: str
    scientific_config_hash: str
    execution_config_hash: str

    def __post_init__(self) -> None:
        _required_text(self.portable_source_yaml, "portable_source_yaml")
        _required_text(self.resolved_config_yaml, "resolved_config_yaml")
        _sha256(self.scientific_config_hash, "scientific_config_hash")
        _sha256(self.execution_config_hash, "execution_config_hash")
