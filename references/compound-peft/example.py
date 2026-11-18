"""
Compound Adapter — Minimal Working Example

Demonstrates how to apply compound adapters to a pretrained model
using the standard PEFT get_peft_model() workflow.

Usage:
    pip install -e peft/
    python example.py
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, CompoundConfig


def main():
    model_name = "microsoft/deberta-v3-base"
    print(f"Loading {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Config 1: Standard compound (C1 + C2) ---
    config_standard = CompoundConfig(
        r=3,
        compound_pattern=["comp_1", "comp_2"],
        compound_type="comp",
        target_modules=["query_proj", "value_proj"],
        use_orthogonal=True,
        init_weights=True,
    )

    peft_model = get_peft_model(model, config_standard)
    print("\n=== Config 1: Standard C1+C2 ===")
    peft_model.print_trainable_parameters()

    # Quick forward pass sanity check
    inputs = tokenizer("Compound adapters are parameter-efficient.", return_tensors="pt")
    outputs = peft_model(**inputs)
    print(f"Forward pass OK — logits shape: {outputs.logits.shape}")

    # --- Config 2: Multi-adapter with offset blocks ---
    del peft_model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    config_offset = CompoundConfig(
        r=3,
        compound_pattern=["comp_1", "comp_2"],
        compound_type="comp",
        target_modules=["query_proj", "value_proj", "key_proj"],
        use_orthogonal=True,
        init_weights=True,
        num_adapters=4,
        adapter_multiplicative=True,
        use_offset_blocks=True,
    )

    peft_model = get_peft_model(model, config_offset)
    print("\n=== Config 2: 4 adapters + offset blocks ===")
    peft_model.print_trainable_parameters()

    outputs = peft_model(**inputs)
    print(f"Forward pass OK — logits shape: {outputs.logits.shape}")

    # --- Config 3: With learnable scaling ---
    del peft_model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    config_scaling = CompoundConfig(
        r=3,
        compound_pattern=["comp_1", "comp_2"],
        compound_type="comp",
        target_modules=["query_proj", "value_proj"],
        use_orthogonal=True,
        init_weights=True,
        use_scaling=True,
    )

    peft_model = get_peft_model(model, config_scaling)
    print("\n=== Config 3: C1+C2 with learnable scaling ===")
    peft_model.print_trainable_parameters()

    outputs = peft_model(**inputs)
    print(f"Forward pass OK — logits shape: {outputs.logits.shape}")


if __name__ == "__main__":
    main()
