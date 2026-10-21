"""Checkpoint the intermediate vision path while its dropout remains disabled."""
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from torch import nn
from timm.models.vision_transformer import VisionTransformer
from quic_petf_vla import inject
from ovlab_benchctl.quic_experiments import resolve_targets


def test_checkpointed_eval_vision_preserves_outputs_gradients_names_and_saves_less(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "external/openvla-quic"))
    from quic_petf.host import enable_gradient_checkpointing, module_inventory

    class Language(nn.Module):
        def gradient_checkpointing_enable(self, **kwargs):
            assert kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Language()
            self.vision_backbone = nn.Module()
            self.vision_backbone.featurizer = VisionTransformer(
                img_size=28, patch_size=14, embed_dim=32, depth=4,
                num_heads=4, mlp_ratio=2, num_classes=0, global_pool="map")

        def forward(self, images):
            return self.vision_backbone.featurizer.get_intermediate_layers(images, n={2})[0]

    torch.manual_seed(54)
    plain = Host().eval()
    targets, _ = resolve_targets(module_inventory(plain), {
        "adapter": {"profile": "QP2", "block_count": 2}, "placement": {"id": "custom"},
        "targets": [{"pattern": r".*\.(?:q|kv|qkv|proj|fc1|fc2)", "policy_active_only": True}],
    }, 4)
    inject(plain, targets)
    checked = copy.deepcopy(plain)
    names = tuple(checked.state_dict())
    assert enable_gradient_checkpointing(checked)["vision_blocks"] == 4
    assert enable_gradient_checkpointing(checked)["vision_blocks"] == 4  # Idempotent.
    assert tuple(checked.state_dict()) == names
    images = torch.randn(4, 3, 28, 28)
    with torch.inference_mode():
        expected = plain(images)
        torch.testing.assert_close(checked(images), expected, rtol=0, atol=0)
    readout = torch.randn_like(expected)

    def backward(model):
        saved = []
        parameter_pointers = {parameter.data_ptr() for parameter in model.parameters()}
        def pack(tensor):
            if tensor.data_ptr() not in parameter_pointers:
                saved.append(tensor.numel() * tensor.element_size())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            values = model(images)
            (values * readout).sum().backward()
        return values.detach(), sum(saved)

    ordinary_values, ordinary_saved = backward(plain)
    checked_values, checked_saved = backward(checked)
    torch.testing.assert_close(checked_values, ordinary_values, rtol=0, atol=0)
    assert checked_saved < ordinary_saved
    for (name, ordinary), (_, checkpointed) in zip(plain.named_parameters(), checked.named_parameters()):
        if ordinary.requires_grad:
            assert ordinary.grad is not None, name
            assert checkpointed.grad is not None, name
            torch.testing.assert_close(checkpointed.grad, ordinary.grad, rtol=1e-5, atol=1e-6)
        else:
            assert ordinary.grad is None and checkpointed.grad is None
