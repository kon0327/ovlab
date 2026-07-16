"""Pure capability negotiation for benchmark and policy adapters."""

from dataclasses import dataclass, field
from enum import Enum

from .contracts.actions import ActionSpec
from .contracts.capabilities import (
    BenchmarkCapabilities,
    ImageObservationSpec,
    PolicyCapabilities,
    ProprioceptiveObservationSpec,
)
from .contracts.errors import ContractCompatibilityError, validation_error
from .contracts.metadata import Metadata, normalize_metadata
from .contracts.signals import SignalAccess


class CompatibilitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    code: str
    severity: CompatibilitySeverity
    path: str
    message: str

    def __post_init__(self) -> None:
        contract = type(self).__name__
        for field_name in ("code", "path", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise validation_error(contract, field_name, "must not be empty or whitespace-only")
        if not isinstance(self.severity, CompatibilitySeverity):
            raise validation_error(contract, "severity", "must be a CompatibilitySeverity")


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]
    benchmark_name: str
    policy_name: str
    contract_version: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = type(self).__name__
        issues = tuple(self.issues)
        if any(not isinstance(issue, CompatibilityIssue) for issue in issues):
            raise validation_error(contract, "issues", "must contain CompatibilityIssue values")
        has_errors = any(issue.severity is CompatibilitySeverity.ERROR for issue in issues)
        if self.compatible == has_errors:
            raise validation_error(contract, "compatible", "must be false exactly when an error issue exists")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, contract))

    def require_compatible(self) -> None:
        errors = [issue for issue in self.issues if issue.severity is CompatibilitySeverity.ERROR]
        if errors:
            summary = "; ".join(f"{issue.code} at {issue.path}: {issue.message}" for issue in errors)
            raise ContractCompatibilityError(summary)


def _issue(code: str, path: str, message: str, severity=CompatibilitySeverity.ERROR) -> CompatibilityIssue:
    return CompatibilityIssue(code, severity, path, message)


def _check_image(
    supplied: ImageObservationSpec,
    required: ImageObservationSpec,
    path: str,
) -> list[CompatibilityIssue]:
    issues = []
    if not set(supplied.shapes) & set(required.shapes):
        issues.append(_issue("OBSERVATION_SHAPE_MISMATCH", f"{path}.shapes", "no permitted image shape overlaps"))
    if supplied.dtype != required.dtype:
        issues.append(
            _issue("OBSERVATION_DTYPE_MISMATCH", f"{path}.dtype", f"{supplied.dtype} cannot satisfy {required.dtype}")
        )
    if not set(supplied.encodings) & set(required.encodings):
        issues.append(_issue("IMAGE_ENCODING_MISMATCH", f"{path}.encodings", "no image encoding overlaps"))
    if not set(supplied.color_spaces) & set(required.color_spaces):
        issues.append(_issue("IMAGE_COLOR_SPACE_MISMATCH", f"{path}.color_spaces", "no color space overlaps"))
    if supplied.maximum_count < required.minimum_count or required.maximum_count < supplied.minimum_count:
        issues.append(_issue("OBSERVATION_COUNT_MISMATCH", f"{path}.count", "image count ranges do not overlap"))
    return issues


def _check_proprioception(
    supplied: ProprioceptiveObservationSpec,
    required: ProprioceptiveObservationSpec,
    path: str,
) -> list[CompatibilityIssue]:
    issues = []
    if not set(supplied.shapes) & set(required.shapes):
        issues.append(
            _issue("OBSERVATION_SHAPE_MISMATCH", f"{path}.shapes", "no permitted proprioceptive shape overlaps")
        )
    if supplied.dtype != required.dtype:
        issues.append(
            _issue("OBSERVATION_DTYPE_MISMATCH", f"{path}.dtype", f"{supplied.dtype} cannot satisfy {required.dtype}")
        )
    if supplied.units != required.units:
        issues.append(_issue("OBSERVATION_UNIT_MISMATCH", f"{path}.units", "proprioceptive units differ"))
    return issues


def _check_action(benchmark: ActionSpec, policy: ActionSpec) -> list[CompatibilityIssue]:
    issues = []
    checks = (
        ("dimension", "ACTION_DIMENSION_MISMATCH"),
        ("representation", "ACTION_REPRESENTATION_MISMATCH"),
        ("translation_indices", "ACTION_TRANSLATION_INDICES_MISMATCH"),
        ("rotation_indices", "ACTION_ROTATION_INDICES_MISMATCH"),
        ("gripper_indices", "ACTION_GRIPPER_INDICES_MISMATCH"),
        ("rotation_representation", "ROTATION_CONVENTION_MISMATCH"),
        ("gripper_convention", "GRIPPER_CONVENTION_MISMATCH"),
        ("units", "ACTION_UNIT_MISMATCH"),
        ("dtype", "ACTION_DTYPE_MISMATCH"),
    )
    for field_name, code in checks:
        if getattr(benchmark, field_name) != getattr(policy, field_name):
            issues.append(_issue(code, f"action_spec.{field_name}", "benchmark and policy declarations differ"))
    return issues


def negotiate_capabilities(
    benchmark: BenchmarkCapabilities,
    policy: PolicyCapabilities,
) -> CompatibilityReport:
    """Return a deterministic report without inserting implicit converters."""
    issues: list[CompatibilityIssue] = []
    if benchmark.contract_version != policy.contract_version:
        issues.append(
            _issue(
                "CONTRACT_VERSION_MISMATCH",
                "contract_version",
                f"benchmark {benchmark.contract_version} != policy {policy.contract_version}",
            )
        )

    supplied_images = {value.name: value for value in benchmark.observation_spec.images}
    supplied_proprioception = {value.name: value for value in benchmark.observation_spec.proprioception}
    signals = {value.name: value for value in benchmark.signal_registry}

    for required in policy.observation_requirements.images:
        path = f"observation_requirements.images.{required.name}"
        supplied = supplied_images.get(required.name)
        if supplied is None:
            signal = signals.get(required.name)
            if signal is not None and signal.access is not SignalAccess.POLICY_VISIBLE:
                issues.append(
                    _issue(
                        "PRIVILEGED_SIGNAL_AS_POLICY_INPUT",
                        path,
                        f"{required.name} is declared {signal.access.value}, not as a policy observation",
                    )
                )
            elif required.required:
                issues.append(_issue("REQUIRED_IMAGE_MISSING", path, "required image is not supplied"))
            else:
                issues.append(
                    _issue(
                        "OPTIONAL_IMAGE_UNAVAILABLE",
                        path,
                        "optional image is not supplied",
                        CompatibilitySeverity.WARNING,
                    )
                )
        else:
            issues.extend(_check_image(supplied, required, path))

    for required in policy.observation_requirements.proprioception:
        path = f"observation_requirements.proprioception.{required.name}"
        supplied = supplied_proprioception.get(required.name)
        if supplied is None:
            signal = signals.get(required.name)
            if signal is not None and signal.access is not SignalAccess.POLICY_VISIBLE:
                issues.append(
                    _issue(
                        "PRIVILEGED_SIGNAL_AS_POLICY_INPUT",
                        path,
                        f"{required.name} is declared {signal.access.value}, not as a policy observation",
                    )
                )
            elif required.required:
                issues.append(_issue("REQUIRED_PROPRIOCEPTION_MISSING", path, "required input is not supplied"))
            else:
                issues.append(
                    _issue(
                        "OPTIONAL_PROPRIOCEPTION_UNAVAILABLE",
                        path,
                        "optional input is not supplied",
                        CompatibilitySeverity.WARNING,
                    )
                )
        else:
            issues.extend(_check_proprioception(supplied, required, path))

    image_count = len(benchmark.observation_spec.images)
    requirements = policy.observation_requirements
    if image_count < requirements.minimum_image_count or (
        requirements.maximum_image_count is not None and image_count > requirements.maximum_image_count
    ):
        issues.append(
            _issue("OBSERVATION_COUNT_MISMATCH", "observation_requirements.image_count", "image count is unsupported")
        )
    proprioception_count = len(benchmark.observation_spec.proprioception)
    if proprioception_count < requirements.minimum_proprioception_count or (
        requirements.maximum_proprioception_count is not None
        and proprioception_count > requirements.maximum_proprioception_count
    ):
        issues.append(
            _issue(
                "OBSERVATION_COUNT_MISMATCH",
                "observation_requirements.proprioception_count",
                "proprioception count is unsupported",
            )
        )

    issues.extend(_check_action(benchmark.action_spec, policy.output_action_spec))
    if benchmark.supports_dynamic_instructions and not policy.supports_dynamic_instructions:
        issues.append(
            _issue(
                "DYNAMIC_INSTRUCTION_UNSUPPORTED",
                "supports_dynamic_instructions",
                "benchmark may revise instructions but policy cannot consume revisions",
            )
        )

    report_version = (
        benchmark.contract_version
        if benchmark.contract_version == policy.contract_version
        else f"{benchmark.contract_version}|{policy.contract_version}"
    )
    return CompatibilityReport(
        compatible=not any(issue.severity is CompatibilitySeverity.ERROR for issue in issues),
        issues=tuple(issues),
        benchmark_name=benchmark.component_name,
        policy_name=policy.component_name,
        contract_version=report_version,
    )
