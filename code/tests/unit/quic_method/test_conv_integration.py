"""Target coverage uses the same module inventory before loading dense weights."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from ovlab_benchctl.quic_experiments import resolve_targets, target_coverage


ROOT = Path(__file__).resolve().parents[4]


def tiny_inventory(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "external/openvla-quic"))
    from quic_petf.host import module_inventory

    # Match the real two-branch vision naming and second-to-last-block contract.
    model = nn.Module()
    model.vision_backbone = nn.Module()
    encoder = model.vision_backbone.featurizer = nn.Module()
    encoder.patch_embed = nn.Module()
    encoder.patch_embed.proj = nn.Conv2d(3, 32, 14, stride=14, device="meta")
    encoder.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    for block in encoder.blocks:
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(32, 96, device="meta")
    model.projector = nn.Module()
    model.projector.fc1 = nn.Linear(32, 32, device="meta")
    model.embedding = nn.Embedding(10, 32, device="meta")
    encoder.attn_pool = nn.Module()
    encoder.attn_pool.q = nn.Linear(32, 32, device="meta")
    encoder.norm = nn.LayerNorm(32, device="meta")
    return module_inventory(model)


def test_meta_inventory_preserves_patch_geometry_and_complete_active_coverage(monkeypatch):
    inventory = tiny_inventory(monkeypatch)
    patch = next(row for row in inventory if row["name"].endswith("patch_embed.proj"))
    assert patch["weight_shape"] == [32, 3, 14, 14]
    assert patch["out_features"] == 32 and patch["module_type"] == "Conv2d"
    config = {"adapter": {"profile": "QP2"}, "placement": {"id": "custom", "action_proximal_last_n": 0},
              "targets": [{"pattern": r"(?:.*\.)?(?:proj|fc1|qkv)", "policy_active_only": True}]}
    targets, manifest = resolve_targets(inventory, config, 2)
    assert len(targets) == 3
    assert {entry["type"] for entry in manifest} == {"Linear", "Conv2d"}
    coverage = target_coverage(inventory, config, targets, 2)
    assert coverage["active_coverage_fraction"] == 1
    assert coverage["uncovered_active_module_count"] == 0
    assert coverage["matched_module_count"] == 4
    assert coverage["excluded_inactive_count"] == 1
    assert coverage["excluded_inactive_modules"][0]["name"].endswith("blocks.1.attn.qkv")
    assert "second-to-last" in coverage["excluded_inactive_modules"][0]["inactive_reason"]
    del config["targets"][0]["policy_active_only"]
    with pytest.raises(ValueError, match="does not contribute"):
        resolve_targets(inventory, config, 2)


def test_intermediate_vision_path_explicitly_excludes_pooling_and_final_norm(monkeypatch):
    inventory = tiny_inventory(monkeypatch)
    config = {"adapter": {"profile": "QP2"}, "placement": {"id": "custom"},
              "targets": [{"pattern": r"vision_backbone\..*\.(?:q|qkv|norm)", "policy_active_only": True}]}
    targets, _ = resolve_targets(inventory, config, 4)
    assert list(targets) == ["vision_backbone.featurizer.blocks.0.attn.qkv"]
    coverage = target_coverage(inventory, config, targets, 4)
    assert coverage["excluded_inactive_count"] == 3
    assert coverage["active_coverage_fraction"] == 1
    for row in inventory:
        if ".attn_pool." in row["name"] or row["name"].endswith("featurizer.norm"):
            assert row["used_in_policy"] is False
            assert "bypass" in row["inactive_reason"]


def test_unsupported_weighted_matches_are_errors_instead_of_silently_excluded(monkeypatch):
    inventory = tiny_inventory(monkeypatch)
    config = {"adapter": {"profile": "QP2"}, "placement": {"id": "custom"},
              "targets": [{"pattern": "embedding"}]}
    with pytest.raises(ValueError, match="Unsupported QuIC target type Embedding"):
        resolve_targets(inventory, config, 4)


def test_grouped_conv_geometry_fails_before_loading_weights(monkeypatch):
    inventory = tiny_inventory(monkeypatch)
    patch = next(row for row in inventory if row["name"].endswith("patch_embed.proj"))
    patch.update(in_features=32, in_channels=32, groups=16)
    config = {"adapter": {"profile": "QP2"}, "placement": {"id": "custom"},
              "targets": [{"names": [patch["name"]]}]}
    with pytest.raises(ValueError, match="block_count must be divisible by groups"):
        resolve_targets(inventory, config, 4)


def test_real_timm_intermediate_path_has_gradients_for_every_selected_target(monkeypatch):
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer
    from quic_petf_vla import adapter_parameters, inject
    monkeypatch.syspath_prepend(str(ROOT / "external/openvla-quic"))
    from quic_petf.host import module_inventory

    model = nn.Module()
    model.vision_backbone = nn.Module()
    encoder = VisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=3,
                                num_heads=4, mlp_ratio=2, num_classes=0, global_pool="map")
    model.vision_backbone.featurizer = encoder
    config = {"adapter": {"profile": "QP2", "block_count": 2}, "placement": {"id": "custom"},
              "targets": [{"pattern": r".*\.(?:q|kv|qkv|proj|fc1|fc2)", "policy_active_only": True}]}
    inventory = module_inventory(model)
    targets, _ = resolve_targets(inventory, config, 4)
    assert len(targets) == 9  # Patch projection and four projections in each of two blocks.
    assert target_coverage(inventory, config, targets, 4)["excluded_inactive_count"] == 9
    inject(model, targets)
    features = encoder.get_intermediate_layers(torch.randn(4, 3, 28, 28), n={1})[0]
    (features * torch.randn_like(features)).sum().backward()
    for parameter in adapter_parameters(model):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
