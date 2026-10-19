"""Numerical validation-objective checks; run in the pinned model environment."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ovlab_benchctl.training_quic import _validation, checked_action_transform


class Tokenizer:
    action_token_begin_idx = 5
    tokenizer = SimpleNamespace(vocab_size=10)

    def decode_token_ids_to_actions(self, ids):
        return np.asarray(ids, dtype=float)


class Model(torch.nn.Module):
    def forward(self, **batch):
        # Three vision logits precede four text positions. Only labels 8 and 9
        # are action targets; arbitrary prompt scores must not enter metrics.
        logits = torch.zeros(1, 7, 10)
        logits[0, 3, 8] = 2
        logits[0, 4, 2] = 2
        logits[0, :3, 0] = 1000
        return SimpleNamespace(logits=logits)


def test_action_diagnostics_align_vision_prefix_and_exclude_prompt_and_eos():
    batch = {"input_ids": torch.tensor([[1, 8, 9, 2]]), "labels": torch.tensor([[-100, 8, 9, -100]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long), "pixel_values": torch.zeros(1, 3, 2, 2)}
    result = _validation(Model(), [batch], Tokenizer(), device="cpu", dtype=torch.float32,
        max_batches=1, description="Objective test")
    assert result["action_tokens"] == 2
    assert result["action_token_accuracy"] == .5
    assert result["invalid_action_token_rate"] == .5
    assert result["decoded_normalized_l1"] == 3.5
    expected = torch.log(torch.exp(torch.tensor(2.)) + 9) - 1
    assert result["action_token_cross_entropy"] == pytest.approx(expected.item())


def test_token_transform_rejects_truncation_and_partial_action_labels():
    tokenizer = SimpleNamespace(padding_side="right", pad_token_id=0, eos_token_id=2,
        model_max_length=16, vocab_size=10)
    example = {"input_ids": torch.tensor([1, 8, 8, 8, 8, 8, 8, 8, 2]),
        "labels": torch.tensor([-100, 8, 8, 8, 8, 8, 8, 8, -100])}
    checked = checked_action_transform(lambda frame: example, tokenizer, Tokenizer())
    assert checked({}) is example
    tokenizer.model_max_length = 8
    with pytest.raises(RuntimeError, match="truncation"):
        checked({})
    tokenizer.model_max_length = 16
    example["labels"][1] = -100
    with pytest.raises(RuntimeError, match="all seven"):
        checked({})
