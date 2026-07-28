# Compound Adapters for PEFT

Parameter-efficient fine-tuning using compound matrix constructions derived from Dynamical Lie Algebra theory. Compound adapters build expressive orthogonal weight matrices from determinants of submatrices, achieving strong performance with very few trainable parameters.

**Paper:** [QuIC: Quantum-Inspired Compound Adapters](https://arxiv.org/abs/2502.06916)

## Installation

```bash
cd peft
pip install -e .
```

This installs a modified version of [HuggingFace PEFT](https://github.com/huggingface/peft) with `PeftType.COMPOUND` added alongside LoRA, OFT, BOFT, etc.

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, CompoundConfig

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

config = CompoundConfig(
    r=3,
    compound_pattern=["comp_1", "comp_2"],
    target_modules=["query_proj", "value_proj"],
    use_orthogonal=True,
    init_weights=True,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~36K  ||  all params: ~86M  ||  trainable%: 0.042%
```

## How It Works

1. A learnable skew-symmetric matrix is mapped to an orthogonal matrix via Cayley parametrization
2. Compound operations (determinants of k×k submatrices) expand the representation
3. Block-diagonal structure assembles compound blocks, padded to match layer dimensions
4. Multiple adapters can be chained multiplicatively for richer transformations

The key insight: the k-th compound of an orthogonal matrix is itself orthogonal in a higher-dimensional space, giving expressiveness without leaving the orthogonal manifold.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 8 | Rank — number of blocks in the block-diagonal structure |
| `compound_pattern` | None | Which compound orders to use, e.g. `["comp_1", "comp_2"]` |
| `compound_type` | `"comp"` | Operation type: `"comp"` (determinant), `"max"`, `"avg"`, `"perm"` |
| `target_modules` | None | Module names or regex to apply adapters to |
| `use_orthogonal` | True | Enforce orthogonality via Cayley parametrization |
| `block_share` | False | Share parameters across diagonal blocks (fewer params) |
| `num_adapters` | 1 | Number of adapter matrices to chain together |
| `adapter_multiplicative` | True | Multiply adapters (True) or add them (False) |
| `use_scaling` | False | Learnable per-dimension scaling after rotation |
| `use_offset_blocks` | False | BOFT-inspired shifted block boundaries for multi-adapter |
| `init_weights` | True | Initialize to identity (True) or random (False) |
| `module_dropout` | 0.0 | Probability of skipping adapter during training |

## Configuration Examples

### Minimal (1st compound only — equivalent to block-diagonal OFT)
```python
CompoundConfig(
    r=3,
    compound_pattern=["comp_1"],
    target_modules=["query_proj", "value_proj"],
)
```

### Standard (1st + 2nd compound)
```python
CompoundConfig(
    r=3,
    compound_pattern=["comp_1", "comp_2"],
    target_modules=["query_proj", "value_proj"],
)
```

### Multi-adapter with offset blocks (BOFT-inspired)
```python
CompoundConfig(
    r=3,
    compound_pattern=["comp_1", "comp_2"],
    target_modules=["query_proj", "value_proj", "key_proj", "intermediate.dense"],
    num_adapters=4,
    use_offset_blocks=True,
)
```

### With learnable scaling
```python
CompoundConfig(
    r=3,
    compound_pattern=["comp_1", "comp_2"],
    target_modules=["query_proj", "value_proj"],
    use_scaling=True,
)
```

## Comparison with Other Adapters

| Method | Weight Structure | Params (DeBERTa-base) | Approach |
|--------|-----------------|----------------------|----------|
| **LoRA** | Low-rank (BA) | ~300K (r=8) | Additive low-rank update |
| **OFT** | Block-diagonal orthogonal | ~36K (r=3) | Cayley on each block |
| **BOFT** | Butterfly orthogonal | ~50K (m=2) | Butterfly factorization |
| **Compound** | Block-diagonal compound | ~36K (r=3, C1+C2) | Compound matrices of orthogonal blocks |

Compound adapters achieve comparable or better accuracy than LoRA with 5-10x fewer parameters, while maintaining orthogonal structure throughout training.

## Citation

```bibtex
@article{raj2025quic,
  title={QuIC: Quantum-Inspired Compound Adapters with Dynamical Lie Algebra},
  author={Raj, Snehal and Music, Luka and Kashefi, Elham},
  journal={arXiv preprint arXiv:2502.06916},
  year={2025}
}
```

## License

Apache 2.0 (same as PEFT)
