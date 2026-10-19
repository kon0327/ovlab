"""OpenVLA stack integration: gradients through checkpointed Llama blocks."""

import pytest

torch = pytest.importorskip("torch")
from quic_petf_vla import AdapterConfig, adapter_parameters, inject


def test_late_llama_adapter_receives_gradients_with_nonreentrant_checkpointing():
    transformers = pytest.importorskip("transformers")
    LlamaConfig, LlamaForCausalLM = transformers.LlamaConfig, transformers.LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                       num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
                                       attention_dropout=0.0, use_cache=False))
    inject(model, {"model.layers.1.self_attn.o_proj": AdapterConfig(
        orders=(1, 2, 3), block_count=1, base_size=3)})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    tokens = torch.randint(0, 64, (2, 7))
    model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()
    parameter = adapter_parameters(model)[0]
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.norm().item() > 1e-8
    assert all(p.grad is None for p in model.parameters() if p is not parameter)

