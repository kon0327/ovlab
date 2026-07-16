"""Capability negotiation and compatibility report tests."""

from dataclasses import replace

import pytest

from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from helpers.mock_specs import mock_action_spec, mock_observation_requirements
from ovlab_core import CompatibilitySeverity, negotiate_capabilities
from ovlab_core.contracts import (
    BenchmarkCapabilities,
    ActionRepresentation,
    ColorSpace,
    ContractCompatibilityError,
    GripperConvention,
    ImageObservationSpec,
    ObservationRequirements,
    PolicyCapabilities,
    ProprioceptiveObservationSpec,
    RotationRepresentation,
    RunContext,
    RunId,
    SignalAccess,
    SignalRegistry,
    SignalSpec,
)


def capability_pair():
    run = RunContext(RunId("run"), 0, "compatibility", 0)
    benchmark = MockBenchmark().initialize(run)
    policy = MockPolicy().initialize(run)
    return benchmark, policy


def issue_codes(benchmark, policy) -> tuple[str, ...]:
    return tuple(issue.code for issue in negotiate_capabilities(benchmark, policy).issues)


def test_valid_compatible_pair() -> None:
    benchmark, policy = capability_pair()
    report = negotiate_capabilities(benchmark, policy)
    assert report.compatible
    assert report.issues == ()
    report.require_compatible()


def test_contract_version_mismatch() -> None:
    benchmark, policy = capability_pair()
    assert "CONTRACT_VERSION_MISMATCH" in issue_codes(benchmark, replace(policy, contract_version="0.2.0"))


def test_missing_required_image() -> None:
    benchmark, policy = capability_pair()
    requirements = replace(policy.observation_requirements, images=())
    missing = replace(mock_observation_requirements().images[0], name="missing")
    requirements = replace(requirements, images=(missing,))
    assert "REQUIRED_IMAGE_MISSING" in issue_codes(benchmark, replace(policy, observation_requirements=requirements))


def test_missing_required_proprioception() -> None:
    benchmark, policy = capability_pair()
    missing = replace(mock_observation_requirements().proprioception[0], name="missing")
    requirements = replace(policy.observation_requirements, proprioception=(missing,))
    assert "REQUIRED_PROPRIOCEPTION_MISSING" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


def test_observation_shape_mismatch() -> None:
    benchmark, policy = capability_pair()
    image = replace(policy.observation_requirements.images[0], shapes=((8, 8, 3),))
    requirements = replace(policy.observation_requirements, images=(image,))
    assert "OBSERVATION_SHAPE_MISMATCH" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


def test_observation_dtype_mismatch() -> None:
    benchmark, policy = capability_pair()
    image = replace(policy.observation_requirements.images[0], dtype="float32")
    requirements = replace(policy.observation_requirements, images=(image,))
    assert "OBSERVATION_DTYPE_MISMATCH" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


def test_color_space_mismatch() -> None:
    benchmark, policy = capability_pair()
    image = replace(policy.observation_requirements.images[0], color_spaces=(ColorSpace.BGR,))
    requirements = replace(policy.observation_requirements, images=(image,))
    assert "IMAGE_COLOR_SPACE_MISMATCH" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_code"),
    [
        ("dimension", 4, "ACTION_DIMENSION_MISMATCH"),
        ("representation", ActionRepresentation.JOINT_DELTA, "ACTION_REPRESENTATION_MISMATCH"),
        ("rotation_representation", RotationRepresentation.AXIS_ANGLE, "ROTATION_CONVENTION_MISMATCH"),
        ("gripper_convention", GripperConvention.CLOSED_POSITIVE, "GRIPPER_CONVENTION_MISMATCH"),
        ("units", ("cm", "cm", "unitless"), "ACTION_UNIT_MISMATCH"),
    ],
)
def test_action_mismatches(field_name, value, expected_code) -> None:
    benchmark, policy = capability_pair()
    action = mock_action_spec()
    if field_name == "dimension":
        action = replace(
            action,
            dimension=4,
            units=("m", "m", "unitless", "unitless"),
            minimum=None,
            maximum=None,
        )
    elif field_name == "rotation_representation":
        action = replace(action, rotation_indices=(1,), translation_indices=(0,), rotation_representation=value)
    else:
        action = replace(action, **{field_name: value})
    report = negotiate_capabilities(benchmark, replace(policy, output_action_spec=action))
    assert expected_code in tuple(issue.code for issue in report.issues)


def test_proprioceptive_unit_mismatch() -> None:
    benchmark, policy = capability_pair()
    proprio = replace(policy.observation_requirements.proprioception[0], units=("degree", "degree"))
    requirements = replace(policy.observation_requirements, proprioception=(proprio,))
    assert "OBSERVATION_UNIT_MISMATCH" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


def test_unsupported_dynamic_instructions() -> None:
    benchmark, policy = capability_pair()
    benchmark = replace(benchmark, supports_dynamic_instructions=True)
    assert "DYNAMIC_INSTRUCTION_UNSUPPORTED" in issue_codes(benchmark, policy)


def test_optional_observation_unavailable_is_warning_only() -> None:
    benchmark, policy = capability_pair()
    optional = replace(policy.observation_requirements.images[0], name="optional", required=False, minimum_count=0)
    requirements = replace(policy.observation_requirements, images=policy.observation_requirements.images + (optional,))
    report = negotiate_capabilities(benchmark, replace(policy, observation_requirements=requirements))
    assert report.compatible
    assert report.issues[0].code == "OPTIONAL_IMAGE_UNAVAILABLE"
    assert report.issues[0].severity is CompatibilitySeverity.WARNING
    report.require_compatible()


def test_privileged_signal_cannot_be_requested_as_policy_input() -> None:
    benchmark, policy = capability_pair()
    hidden = replace(policy.observation_requirements.images[0], name="hidden_target")
    requirements = replace(policy.observation_requirements, images=(hidden,))
    assert "PRIVILEGED_SIGNAL_AS_POLICY_INPUT" in issue_codes(
        benchmark, replace(policy, observation_requirements=requirements)
    )


def test_issue_codes_and_order_are_stable() -> None:
    benchmark, policy = capability_pair()
    missing_image = replace(policy.observation_requirements.images[0], name="missing-image")
    missing_proprio = replace(policy.observation_requirements.proprioception[0], name="missing-state")
    requirements = replace(policy.observation_requirements, images=(missing_image,), proprioception=(missing_proprio,))
    incompatible = replace(policy, contract_version="x", observation_requirements=requirements)
    first = issue_codes(benchmark, incompatible)
    second = issue_codes(benchmark, incompatible)
    assert first == second
    assert first[:3] == (
        "CONTRACT_VERSION_MISMATCH",
        "REQUIRED_IMAGE_MISSING",
        "REQUIRED_PROPRIOCEPTION_MISSING",
    )


def test_require_compatible_raises_only_for_errors() -> None:
    benchmark, policy = capability_pair()
    incompatible = replace(policy, contract_version="x")
    with pytest.raises(ContractCompatibilityError, match="CONTRACT_VERSION_MISMATCH"):
        negotiate_capabilities(benchmark, incompatible).require_compatible()
