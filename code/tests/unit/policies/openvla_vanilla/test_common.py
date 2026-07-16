from dataclasses import replace
import sys

import numpy as np
import pytest

from ovlab_core.contracts import GripperConvention
from ovlab_openvla_common import (
    LiberoActionCodec, OpenVlaActionCodecError, OpenVlaDecodedAction,
    OpenVlaModelSource, OpenVlaPromptFormatter, libero_target_action_spec,
)
from ovlab_openvla_vanilla import ModelDType, OpenVlaVanillaSettings


def test_exact_prompt_regression_and_empty_rejection():
    formatter = OpenVlaPromptFormatter()
    assert formatter.format("Put The Object In The Basket") == (
        "In: What action should the robot take to put the object in the basket?\nOut:"
    )
    assert formatter.identifier == "openvla-v1@1.0.0"
    with pytest.raises(ValueError):
        formatter.format("  ")


@pytest.mark.parametrize("source,expected", [(0.0, 1.0), (1.0, -1.0), (0.5, 0.0), (0.49, 1.0), (0.51, -1.0)])
def test_exact_gripper_normalization_binarization_and_inversion(source, expected):
    pose = np.array([-.9, -.5, 0, .25, .5, .9], dtype=np.float64)
    result = LiberoActionCodec().encode(OpenVlaDecodedAction(np.r_[pose, source]))
    np.testing.assert_array_equal(result[:6], pose.astype(np.float32))
    assert result[6] == expected
    assert result.dtype == np.float32 and not result.flags.writeable


def test_codec_validates_ranges_and_typed_stage_prevents_double_conversion():
    codec = LiberoActionCodec()
    converted = codec.encode(OpenVlaDecodedAction(np.array([0, 0, 0, 0, 0, 0, 1])))
    with pytest.raises(OpenVlaActionCodecError, match="only OpenVlaDecodedAction"):
        codec.encode(converted)
    with pytest.raises(OpenVlaActionCodecError, match="source"):
        codec.encode(OpenVlaDecodedAction(np.array([0, 0, 0, 0, 0, 0, -1])))
    with pytest.raises(OpenVlaActionCodecError, match="pose"):
        codec.encode(OpenVlaDecodedAction(np.array([2, 0, 0, 0, 0, 0, 1])))


def test_settings_are_immutable_validated_and_hash_deterministic(tmp_path):
    source = OpenVlaModelSource(str(tmp_path), revision="abc")
    first = OpenVlaVanillaSettings(source, "bridge_orig", metadata={"b": 2, "a": (1,)})
    second = OpenVlaVanillaSettings(source, "bridge_orig", metadata={"a": [1], "b": 2})
    assert first.settings_hash == second.settings_hash
    assert first.local_files_only and first.model_dtype is ModelDType.BFLOAT16
    with pytest.raises(Exception):
        first.device = "cpu"
    wrong = replace(libero_target_action_spec(), gripper_convention=GripperConvention.OPEN_POSITIVE)
    with pytest.raises(ValueError, match="incompatible"):
        OpenVlaVanillaSettings(source, "bridge_orig", target_action_spec=wrong)


def test_import_is_torch_free():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
