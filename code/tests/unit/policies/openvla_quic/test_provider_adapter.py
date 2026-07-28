from dataclasses import replace
import importlib
import sys

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from ovlab_core.contracts import (
    AdapterState,
    ColorSpace,
    ImageEncoding,
    ImageObservation,
    ImageObservationSpec,
    Instruction,
    InstructionId,
    InstructionSource,
    PolicyObservation,
    ProprioceptiveObservation,
    ProprioceptiveObservationSpec,
    StepId,
)
from ovlab_openvla_common import libero_target_action_spec
from ovlab_openvla_quic import (
    OpenVLAQuICPEFTAdapter,
    OpenVLAQuICWCAdapter,
    QuICImplementationStatus,
    QuICImplementationUnavailableError,
    QuICPEFTIntegrationIncompleteError,
    QuICProfileDefinition,
    QuICProfileId,
    QuICProviderContractError,
    QuICProviderLoader,
    QuICProviderSpec,
    QuICVariant,
    QuICWCImplementationIncompleteError,
    create_runtime_adapter,
    skeleton_descriptor,
)
from ovlab_remote_policy.protocol import (
    action_spec_to_wire,
    image_spec_to_wire,
    proprio_spec_to_wire,
)


def _available(identity: str) -> dict[str, object]:
    return {"availability": "available", "identity": identity}


def _runtime_descriptor(variant: QuICVariant):
    if variant is QuICVariant.PEFT:
        artifact = {"availability": "available", "form": "multiplicative_adapter", "identity": "adapter@test"}
        deployment = {
            "availability": "available", "active_adapter": True, "merge_state": "unmerged",
            "requires_base_model": True, "deployment_replaces_base_weights": False,
        }
    else:
        artifact = {"availability": "available", "form": "compact_weight_factors", "identity": "factors@test"}
        deployment = {
            "availability": "available", "replacement_state": "replaced",
            "requires_replaced_dense_weights_at_deployment": False,
            "deployment_replaces_selected_weights": True,
            "dense_runtime_reconstruction_allowed": False,
        }
    return replace(
        skeleton_descriptor(variant),
        implementation_status=QuICImplementationStatus.IMPLEMENTED,
        source_import_status="present",
        openvla_integration_status="implemented",
        source_identity=(
            skeleton_descriptor(variant).source_identity
            if variant is QuICVariant.PEFT
            else _available("wc-source@test")
        ),
        profile=QuICProfileDefinition(QuICProfileId.QP0),
        base_model_identity=_available("base@test"),
        artifact_identity=artifact,
        provenance_identity=_available(f"provenance@{variant.value}"),
        deployment_state=deployment,
        capability_identity=_available("capabilities@test"),
        normalization_identity=_available("normalization@test"),
        parameterization=_available("parameters@test"),
    )


PRIMARY_SPEC = ImageObservationSpec(
    name="camera.primary.rgb",
    shapes=((2, 2, 3),),
    dtype="uint8",
    encodings=(ImageEncoding.RAW,),
    color_spaces=(ColorSpace.RGB,),
)
WRIST_SPEC = ImageObservationSpec(
    name="camera.wrist.rgb",
    shapes=((2, 2, 3),),
    dtype="uint8",
    encodings=(ImageEncoding.RAW,),
    color_spaces=(ColorSpace.RGB,),
)
PROPRIO_SPEC = ProprioceptiveObservationSpec(
    name="proprioception.joint_state",
    shapes=((3,),),
    dtype="float32",
    units=("rad", "rad", "unitless"),
)


class ExplicitTestDoubleQuICProvider:
    """Test-only neutral provider; never registered or configurable in production."""

    def __init__(self, descriptor, horizon: int) -> None:
        self.descriptor = descriptor
        variant = descriptor.variant
        self.variant = variant
        self.horizon = horizon
        self.requests = []
        self.load_requests = []
        self.reset_requests = []
        self.closed = False
        self.loads = {"base_model": 1, "quic_artifact": 1, "processor": 1}

    def api_version(self):
        return "1.0.0"

    def describe(self):
        expected = self.descriptor.canonical_dict()
        return {
            "family": expected["family"],
            "variant_id": expected["variant_id"],
            "scientific_identity_hash": self.descriptor.scientific_hash,
            "base_model_identity": expected["base_model_identity"],
            "artifact_identity": expected["artifact_identity"],
            "normalization_identity": expected["normalization_identity"],
            "profile": expected["profile"],
            "placement_manifest": expected["placement_manifest"],
        }

    def capability_description(self):
        images = (PRIMARY_SPEC,) if self.horizon == 1 else (PRIMARY_SPEC, WRIST_SPEC)
        proprio = () if self.horizon == 1 else (PROPRIO_SPEC,)
        return {
            "images": [image_spec_to_wire(item) for item in images],
            "proprioception": [proprio_spec_to_wire(item) for item in proprio],
            "action_spec": action_spec_to_wire(libero_target_action_spec()),
            "minimum_horizon": self.horizon,
            "maximum_horizon": self.horizon,
            "dynamic_instructions": True,
            "deterministic_reset": True,
        }

    def load(self, request):
        self.load_requests.append(request)
        return {"model": "explicit-test-double", "artifact": self.variant.value}

    def reset_episode(self, request):
        self.reset_requests.append(request)

    def predict(self, request):
        self.requests.append(request)
        actions = np.zeros((self.horizon, 7), dtype=np.float32)
        actions[:, 6] = 1.0
        return {
            "actions": actions,
            "inference_duration_ns": 20,
            "metadata": {"test_double": True},
        }

    def close(self):
        self.closed = True

    def load_counts(self):
        return self.loads


def _observation(*, multimodal: bool) -> PolicyObservation:
    instruction = Instruction(
        InstructionId("instruction-0"),
        "move the object",
        10,
        InstructionSource.BENCHMARK,
    )
    images = [
        ImageObservation(
            "camera.primary.rgb",
            np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            10,
            ImageEncoding.RAW,
            ColorSpace.RGB,
            "agentview",
        )
    ]
    proprio = []
    if multimodal:
        images.append(ImageObservation(
            "camera.wrist.rgb",
            np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
            10,
            ImageEncoding.RAW,
            ColorSpace.RGB,
            "robot0_eye_in_hand",
        ))
        proprio.append(ProprioceptiveObservation(
            "proprioception.joint_state",
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            10,
            ("rad", "rad", "unitless"),
        ))
    return PolicyObservation(StepId("episode-0-step-0"), 10, instruction, tuple(images), tuple(proprio))


class _Clock:
    def __init__(self):
        self.values = iter((100, 200))

    def __call__(self):
        return next(self.values)


@pytest.mark.parametrize(
    ("variant", "adapter_type", "horizon"),
    (
        (QuICVariant.PEFT, OpenVLAQuICPEFTAdapter, 1),
        (QuICVariant.WC, OpenVLAQuICWCAdapter, 3),
    ),
)
def test_adapter_delegates_single_and_chunk_protocol_without_privileged_inputs(
    variant, adapter_type, horizon
):
    descriptor = _runtime_descriptor(variant)
    provider = ExplicitTestDoubleQuICProvider(descriptor, horizon)
    adapter = adapter_type(
        descriptor,
        _test_provider=provider,
        clock_ns=_Clock(),
        wall_clock_ns=lambda: 300,
    )
    run, episode = make_run_context(), make_episode_context()
    capabilities = adapter.initialize(run)
    assert capabilities.supports_single_action is (horizon == 1)
    assert capabilities.supports_action_chunks is (horizon > 1)
    assert capabilities.minimum_action_horizon == horizon
    assert capabilities.maximum_action_horizon == horizon
    assert action_spec_to_wire(capabilities.output_action_spec) == action_spec_to_wire(
        libero_target_action_spec()
    )
    assert capabilities.metadata["method_descriptor"]["variant_id"] == variant.value
    assert capabilities.metadata["provider_identity"]["test_double"] is True

    adapter.reset_episode(episode)
    prediction = adapter.predict(_observation(multimodal=horizon > 1))
    assert prediction.actions.shape == (horizon, 7)
    assert prediction.actions.dtype == np.float32
    assert prediction.horizon == horizon
    assert prediction.metadata["model_inference_duration_ns"] == 20
    assert prediction.inference_duration_ns == 100

    request = provider.requests[0]
    assert set(request) == {
        "request_id", "episode_id", "step_id", "instruction", "images", "proprioception"
    }
    forbidden = {
        "reward", "success", "goal_predicate", "collision", "object_poses",
        "simulator", "future_observations", "termination_state", "contacts",
    }
    assert not forbidden & set(request)
    assert request["instruction"] == "move the object"
    assert np.array_equal(
        request["images"]["camera.primary.rgb"]["data"],
        _observation(multimodal=horizon > 1).images[0].data,
    )
    assert request["images"]["camera.primary.rgb"]["layout"] == "HWC"

    adapter.end_episode(episode)
    adapter.close()
    assert provider.closed is True
    assert adapter.state is AdapterState.CLOSED


class ExplodingTestLoader:
    def __init__(self):
        self.calls = 0

    def load(self, descriptor):
        del descriptor
        self.calls += 1
        raise AssertionError("provider discovery must not occur for a skeleton")


@pytest.mark.parametrize(
    ("variant", "adapter_type", "next_gate", "error_type", "status", "source"),
    (
        (
            QuICVariant.PEFT, OpenVLAQuICPEFTAdapter, "I",
            QuICPEFTIntegrationIncompleteError,
            "legacy_reference_available_openvla_integration_skeleton",
            "external/openvla-quic -> external/compound-peft",
        ),
        (
            QuICVariant.WC, OpenVLAQuICWCAdapter, "J",
            QuICWCImplementationIncompleteError,
            "source_absent_implementation_skeleton",
            "external/openvla-quic",
        ),
    ),
)
def test_skeleton_fails_typed_before_discovery_cuda_model_checkpoint_socket_or_trace(
    tmp_path, variant, adapter_type, next_gate, error_type, status, source
):
    loader = ExplodingTestLoader()
    adapter = adapter_type(skeleton_descriptor(variant), provider_loader=loader)
    torch_before = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    socket_path = tmp_path / "must-not-exist.sock"
    with pytest.raises(error_type) as failure:
        adapter.initialize(make_run_context())
    assert failure.value.variant == variant.value
    assert failure.value.expected_package == "openvla_quic.ovlab_provider"
    assert failure.value.expected_source == source
    assert failure.value.implementation_status == status
    assert failure.value.next_implementation_gate == next_gate
    assert loader.calls == 0
    assert adapter.state is AdapterState.CREATED
    assert not socket_path.exists()
    assert torch_before == {
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    }

    with pytest.raises(error_type):
        create_runtime_adapter(skeleton_descriptor(variant))


def test_lazy_provider_import_happens_only_after_runtime_descriptor_is_complete(monkeypatch):
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    descriptor = replace(
        descriptor,
        provider=QuICProviderSpec(package="explicitly_absent_test_provider"),
    )
    calls = []
    real_import = importlib.import_module

    def recording_import(name, package=None):
        calls.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", recording_import)
    loader = QuICProviderLoader()
    assert calls == []
    with pytest.raises(QuICImplementationUnavailableError) as failure:
        loader.load(descriptor)
    assert calls == ["explicitly_absent_test_provider"]
    assert failure.value.expected_package == "explicitly_absent_test_provider"


@pytest.mark.parametrize(
    ("actions", "message"),
    (
        (np.zeros((1, 6), dtype=np.float32), "shape"),
        (np.zeros((1, 7), dtype=np.float64), "shape"),
        (np.full((1, 7), np.nan, dtype=np.float32), "finite"),
        (np.full((1, 7), 2.0, dtype=np.float32), "bounds"),
    ),
)
def test_provider_cannot_return_dummy_or_contract_incompatible_actions(actions, message):
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    provider = ExplicitTestDoubleQuICProvider(descriptor, 1)
    provider.predict = lambda request: {
        "actions": actions,
        "inference_duration_ns": 20,
        "metadata": {},
    }
    adapter = OpenVLAQuICPEFTAdapter(
        descriptor,
        _test_provider=provider,
        clock_ns=_Clock(),
    )
    adapter.initialize(make_run_context())
    adapter.reset_episode(make_episode_context())
    with pytest.raises(QuICProviderContractError, match=message):
        adapter.predict(_observation(multimodal=False))


def test_provider_identity_mismatch_closes_provider_without_fallback():
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    provider = ExplicitTestDoubleQuICProvider(descriptor, 1)
    provider.describe = lambda: {
        **ExplicitTestDoubleQuICProvider(descriptor, 1).describe(),
        "variant_id": "quic-wc",
    }
    adapter = OpenVLAQuICPEFTAdapter(
        descriptor, _test_provider=provider
    )
    with pytest.raises(QuICProviderContractError, match="identity differs"):
        adapter.initialize(make_run_context())
    assert provider.closed is True
    assert adapter.state is AdapterState.CREATED


def test_provider_capability_schema_rejects_arbitrary_privileged_fields():
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    provider = ExplicitTestDoubleQuICProvider(descriptor, 1)
    original = provider.capability_description
    provider.capability_description = lambda: {
        **original(),
        "reward": True,
    }
    adapter = OpenVLAQuICPEFTAdapter(
        descriptor, _test_provider=provider
    )
    with pytest.raises(QuICProviderContractError, match="capability fields must equal"):
        adapter.initialize(make_run_context())
    assert provider.closed is True


def test_runtime_observation_must_match_negotiated_metadata_before_delegation():
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    provider = ExplicitTestDoubleQuICProvider(descriptor, 1)
    adapter = OpenVLAQuICPEFTAdapter(descriptor, _test_provider=provider)
    adapter.initialize(make_run_context())
    adapter.reset_episode(make_episode_context())
    valid = _observation(multimodal=False)
    invalid_image = ImageObservation(
        "camera.primary.rgb",
        np.zeros((3, 3, 3), dtype=np.uint8),
        10,
        ImageEncoding.RAW,
        ColorSpace.RGB,
        "agentview",
    )
    invalid = PolicyObservation(
        valid.step_id, valid.timestamp_ns, valid.instruction, (invalid_image,)
    )
    with pytest.raises(QuICProviderContractError, match="negotiated metadata"):
        adapter.predict(invalid)
    assert provider.requests == []


def test_provider_prediction_metadata_rejects_privileged_evaluation_fields():
    descriptor = _runtime_descriptor(QuICVariant.PEFT)
    provider = ExplicitTestDoubleQuICProvider(descriptor, 1)
    provider.predict = lambda request: {
        "actions": np.zeros((1, 7), dtype=np.float32),
        "inference_duration_ns": 20,
        "metadata": {"diagnostics": {"success": True}},
    }
    adapter = OpenVLAQuICPEFTAdapter(
        descriptor, _test_provider=provider, clock_ns=_Clock()
    )
    adapter.initialize(make_run_context())
    adapter.reset_episode(make_episode_context())
    with pytest.raises(QuICProviderContractError, match="privileged evaluation"):
        adapter.predict(_observation(multimodal=False))
