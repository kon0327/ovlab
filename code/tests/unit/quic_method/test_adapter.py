"""Mathematical and artifact checks, runnable without external test packages."""

import copy
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="QuIC numerical checks need the pinned host runtime")
from torch import nn
from torch.func import functional_call

from quic_petf_vla import (
    AdapterConfig, CompoundOperator, OrthogonalBase, QuICLinear,
    adapter_manifest, adapter_parameters, compound_matrix, disabled_adapters,
    inject, load_adapter, merge_adapters, save_adapter,
)


HOST = {"checkpoint_id": "unit-test-host", "revision": "a" * 40,
        "checkpoint_sha256": "b" * 64, "adaptation_state": "merged", "family": "test_dense_model"}


def reference_compound(base, order):
    combinations = list(itertools.combinations(range(base.shape[-1]), order))
    return torch.stack([torch.stack([torch.linalg.det(base[..., list(rows), :][..., list(cols)])
                                    for cols in combinations], dim=-1)
                        for rows in combinations], dim=-2)


class CompoundTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_polynomial_values_match_all_minors(self):
        base = torch.randn(2, 4, 4, dtype=torch.float64)
        for order in (1, 2, 3):
            with self.subTest(order=order):
                torch.testing.assert_close(compound_matrix(base, order), reference_compound(base, order), rtol=1e-13, atol=1e-13)

    def test_compound_gradcheck_at_identity_and_random_base(self):
        for order in (1, 2, 3):
            for initialization in ("identity", "random"):
                base = (torch.eye(4, dtype=torch.float64) if initialization == "identity"
                        else torch.randn(4, 4, dtype=torch.float64)).requires_grad_()
                with self.subTest(order=order, initialization=initialization):
                    self.assertTrue(torch.autograd.gradcheck(lambda a: compound_matrix(a, order), (base,),
                                                            eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_cayley_and_compounds_are_orthogonal(self):
        module = OrthogonalBase(5, 2, device=torch.device("cpu"), dtype=torch.float64)
        with torch.no_grad():
            module.skew_entries.normal_(std=0.25)
        matrices = module(torch.float64)
        for order in (1, 2, 3):
            compound = compound_matrix(matrices, order)
            identity = torch.eye(math.comb(5, order), dtype=torch.float64).expand(2, -1, -1)
            torch.testing.assert_close(compound.transpose(-1, -2) @ compound, identity, atol=1e-13, rtol=1e-13)

    def test_adapter_gradcheck_zero_entries(self):
        module = CompoundOperator(28, AdapterConfig(orders=(1, 2, 3), block_count=2, base_size=4,
                                  share_across_blocks=True, construction_dtype="float64"), device=torch.device("cpu"))
        values = torch.randn(2, 28, dtype=torch.float64)
        entries = module.skew_entries.detach().clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda theta: functional_call(module, {"base.skew_entries": theta}, (values,)),
            (entries,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_invalid_orders_and_block_configurations(self):
        for args in ({"orders": ()}, {"orders": (2, 1)}, {"orders": (1, 1)}, {"orders": (4,)},
                     {"orders": (True,)}, {"block_count": 0}, {"base_size": 1},
                     {"orientation": "input"}, {"construction_dtype": "bfloat16"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                AdapterConfig(**args)
        with self.assertRaises(ValueError):
            AdapterConfig(block_count=3).resolve(8)
        with self.assertRaises(ValueError):
            AdapterConfig(orders=(1, 2, 3), block_count=2).resolve(6)
        with self.assertRaises(ValueError):
            AdapterConfig(orders=(1, 2), block_count=2, base_size=5).resolve(16)

    def test_auto_base_size_matches_documented_examples(self):
        for orders, n, padding, count in (((1,), 512, 0, 1046528), ((1, 2), 31, 16, 3720),
                                         ((1, 2, 3), 14, 43, 728)):
            resolved = AdapterConfig(orders=orders, block_count=8).resolve(4096)
            self.assertEqual(resolved.base_size, n)
            self.assertEqual(512 - sum(math.comb(n, k) for k in orders), padding)
            self.assertEqual(8 * n * (n - 1) // 2, count)


class LinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identity_rectangular_linear_and_bias(self):
        for bias in (True, False):
            layer = nn.Linear(7, 32, bias=bias, dtype=torch.float64)
            values = torch.randn(2, 3, 7, dtype=torch.float64)
            expected = layer(values)
            wrapped = QuICLinear(layer, AdapterConfig(orders=(1, 2, 3), block_count=2,
                                 base_size=4, construction_dtype="float64"))
            torch.testing.assert_close(wrapped(values), expected, atol=1e-14, rtol=1e-14)

    def test_bfloat16_identity_reproduces_fused_host_bias_rounding(self):
        layer = nn.Linear(7, 32, dtype=torch.bfloat16)
        values = torch.randn(2, 3, 7, dtype=torch.bfloat16)
        expected = layer(values)
        wrapped = QuICLinear(layer, AdapterConfig(orders=(1, 2, 3), block_count=2, base_size=4))
        torch.testing.assert_close(wrapped(values), expected, atol=0, rtol=0)

    def test_nonidentity_output_orientation_bias_padding_and_gram(self):
        layer = nn.Linear(7, 32, dtype=torch.float64)
        wrapped = QuICLinear(layer, AdapterConfig(orders=(1, 2, 3), block_count=2,
                             base_size=4, construction_dtype="float64"))
        with torch.no_grad():
            wrapped.adapter.skew_entries.normal_(std=0.2)
        matrices = wrapped.adapter.matrices()
        blocks = [torch.block_diag(*(matrix[i] for matrix in matrices), torch.eye(2, dtype=torch.float64))
                  for i in range(2)]
        full = torch.block_diag(*blocks)
        values = torch.randn(2, 3, 7, dtype=torch.float64)
        expected = torch.nn.functional.linear(values, full @ layer.weight, layer.bias)
        torch.testing.assert_close(wrapped(values), expected, atol=1e-13, rtol=1e-13)
        effective = full @ layer.weight
        torch.testing.assert_close(effective.T @ effective, layer.weight.T @ layer.weight, atol=1e-13, rtol=1e-13)
        zero = torch.zeros(7, dtype=torch.float64)
        torch.testing.assert_close(wrapped(zero), layer.bias, atol=0, rtol=0)
        projected = torch.nn.functional.linear(values, layer.weight).reshape(2, 3, 2, 16)
        adapted = wrapped.adapter(projected.flatten(-2)).reshape_as(projected)
        torch.testing.assert_close(adapted[..., -2:], projected[..., -2:], atol=0, rtol=0)

    def test_shared_bases_and_exact_parameter_counts(self):
        for sharing, expected_count in ((True, 6), (False, 12)):
            model = nn.Sequential(nn.Linear(7, 32))
            manifest = inject(model, {"0": AdapterConfig(orders=(1, 2, 3), block_count=2,
                                     base_size=4, share_across_blocks=sharing)})
            self.assertEqual(sum(p.numel() for p in adapter_parameters(model)), expected_count)
            self.assertEqual(manifest[0]["independent_parameters"], expected_count)
            self.assertEqual(manifest[0]["identity_padding_per_block"], 2)
            if sharing:
                with torch.no_grad():
                    model[0].adapter.parameter.normal_(std=0.2)
                repeated = torch.randn(3, 16).repeat(1, 2)
                output = model[0].adapter(repeated)
                torch.testing.assert_close(output[:, :16], output[:, 16:], atol=0, rtol=0)

    def test_gradient_flows_through_frozen_downstream_layers(self):
        model = nn.Sequential(nn.Linear(5, 16, dtype=torch.float64), nn.Tanh(),
                              nn.Linear(16, 3, dtype=torch.float64))
        backbone = {name: p.detach().clone() for name, p in model.named_parameters()}
        inject(model, {"0": AdapterConfig(orders=(1, 2), block_count=2, base_size=3, construction_dtype="float64")})
        initial = model[0].adapter.parameter.detach().clone()
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-2)
        values, desired = torch.randn(4, 5, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
        loss = (model(values) - desired).square().mean()
        loss.backward()
        grad = model[0].adapter.parameter.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 1e-8)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if "adapter" not in name))
        optimizer.step()
        self.assertFalse(torch.equal(initial, model[0].adapter.parameter))
        torch.testing.assert_close(model[0].weight, backbone["0.weight"], atol=0, rtol=0)
        torch.testing.assert_close(model[0].bias, backbone["0.bias"], atol=0, rtol=0)
        torch.testing.assert_close(model[2].weight, backbone["2.weight"], atol=0, rtol=0)

    def test_late_llama_adapter_receives_gradients_with_nonreentrant_checkpointing(self):
        from transformers import LlamaConfig, LlamaForCausalLM
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
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.norm().item(), 1e-8)
        self.assertTrue(all(p.grad is None for p in model.parameters() if p is not parameter))

    def test_injection_validation_is_atomic(self):
        model = nn.Sequential(nn.Linear(3, 16), nn.Linear(16, 4))
        with self.assertRaises(ValueError):
            inject(model, {"0": AdapterConfig(block_count=2, base_size=3), "missing": AdapterConfig()})
        self.assertIs(type(model[0]), nn.Linear)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        with self.assertRaises(TypeError):
            inject(model, {"0": {"block_count": 2}})

    def test_quantized_or_specialized_subclasses_are_rejected(self):
        class SpecializedLinear(nn.Linear):
            pass
        with self.assertRaises(TypeError):
            QuICLinear(SpecializedLinear(2, 16), AdapterConfig(block_count=2, base_size=3))

    def test_disabled_context_restores_baseline_and_nested_state(self):
        original = nn.Sequential(nn.Linear(4, 16, dtype=torch.float64))
        model = copy.deepcopy(original)
        inject(model, {"0": AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")})
        with torch.no_grad():
            model[0].adapter.parameter.normal_(std=0.2)
        values = torch.randn(3, 4, dtype=torch.float64)
        adapted = model(values)
        with disabled_adapters(model):
            torch.testing.assert_close(model(values), original(values), atol=0, rtol=0)
            with disabled_adapters(model):
                self.assertFalse(model[0].adapter_enabled)
            self.assertFalse(model[0].adapter_enabled)
        self.assertTrue(model[0].adapter_enabled)
        torch.testing.assert_close(model(values), adapted, atol=0, rtol=0)

    def test_unconstrained_ablation_uses_full_delta_and_identity(self):
        layer = nn.Linear(4, 16, dtype=torch.float64)
        values = torch.randn(3, 4, dtype=torch.float64)
        expected = layer(values)
        model = nn.Sequential(layer)
        manifest = inject(model, {"0": AdapterConfig(block_count=2, base_size=3,
                                orthogonal=False, construction_dtype="float64")})
        self.assertEqual(manifest[0]["independent_parameters"], 18)
        self.assertEqual(model[0].adapter.parameter.shape, (2, 3, 3))
        torch.testing.assert_close(model(values), expected, atol=1e-14, rtol=1e-14)
        model(values).sum().backward()
        self.assertGreater(model[0].adapter.parameter.grad.norm().item(), 0)

    def test_host_dtype_conversion_preserves_construction_precision(self):
        wrapped = QuICLinear(nn.Linear(4, 16), AdapterConfig(block_count=2, base_size=3))
        wrapped.to(dtype=torch.bfloat16)
        self.assertEqual(wrapped.weight.dtype, torch.bfloat16)
        self.assertEqual(wrapped.adapter.parameter.dtype, torch.float32)
        self.assertEqual(wrapped(torch.randn(2, 4, dtype=torch.bfloat16)).dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_gradient_and_device_transfer(self):
        model = nn.Sequential(nn.Linear(4, 32, dtype=torch.bfloat16, device="cuda"))
        inject(model, {"0": AdapterConfig(orders=(1, 2, 3), block_count=2, base_size=4)})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16))
            output.float().square().mean().backward()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(model[0].adapter.parameter.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(model[0].adapter.parameter.grad).all())
        model.to("cpu")
        self.assertEqual(model[0].adapter.parameter.device.type, "cpu")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.original = nn.Sequential(nn.Linear(4, 16, dtype=torch.float64), nn.Tanh(),
                                      nn.Linear(16, 6, dtype=torch.float64))
        self.model = copy.deepcopy(self.original)
        inject(self.model, {"0": AdapterConfig(block_count=2, base_size=3, construction_dtype="float64")})
        with torch.no_grad():
            self.model[0].adapter.parameter.normal_(std=0.2)
        self.values = torch.randn(2, 3, 4, dtype=torch.float64)

    def test_roundtrip_minimal_tensors_and_merge(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            report = save_adapter(self.model, tmp, HOST, metadata={"step": 3})
            tensors = load_file(str(Path(tmp) / "adapter.safetensors"))
            self.assertEqual(sum(p.numel() for p in tensors.values()), 6)
            self.assertEqual(report["metadata"], {"step": 3})
            restored = copy.deepcopy(self.original)
            manifest = load_adapter(restored, tmp, HOST)
            self.assertEqual(manifest, adapter_manifest(self.model))
            torch.testing.assert_close(restored(self.values), self.model(self.values), atol=0, rtol=0)
            expected = restored(self.values)
            bias = restored[0].bias.detach().clone()
            merge_adapters(restored)
            self.assertIs(type(restored[0]), nn.Linear)
            self.assertEqual(len(adapter_parameters(restored)), 0)
            torch.testing.assert_close(restored(self.values), expected, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(restored[0].bias, bias, atol=0, rtol=0)
            with self.assertRaises(FileExistsError):
                save_adapter(self.model, tmp, HOST)

    def test_host_mismatch_and_payload_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(self.model, tmp, HOST)
            fresh = copy.deepcopy(self.original)
            with self.assertRaisesRegex(ValueError, "host identity"):
                load_adapter(fresh, tmp, {**HOST, "checkpoint_sha256": "c" * 64})
            self.assertIs(type(fresh[0]), nn.Linear)
            self.assertTrue(all(p.requires_grad for p in fresh.parameters()))
            path = Path(tmp) / "adapter.safetensors"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_adapter(fresh, tmp, HOST)

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(self.model, tmp, HOST)
            path = Path(tmp) / "adapter.json"
            report = json.loads(path.read_text())
            report["manifest"][0]["config"]["orders"] = [1]
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "binding"):
                load_adapter(copy.deepcopy(self.original), tmp, HOST)

    def test_host_shape_mismatch_rejected_before_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(self.model, tmp, HOST)
            bad_host = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 6))
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_adapter(bad_host, tmp, HOST)
            self.assertIs(type(bad_host[0]), nn.Linear)

    def test_missing_immutable_host_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                save_adapter(self.model, tmp, {"checkpoint_id": "floating-main"})

    def test_nonorthogonal_roundtrip(self):
        model = copy.deepcopy(self.original)
        inject(model, {"0": AdapterConfig(block_count=2, base_size=3, orthogonal=False,
                                         construction_dtype="float64")})
        with torch.no_grad():
            model[0].adapter.parameter.normal_(std=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(model, tmp, HOST)
            restored = copy.deepcopy(self.original)
            load_adapter(restored, tmp, HOST)
            torch.testing.assert_close(restored(self.values), model(self.values), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
