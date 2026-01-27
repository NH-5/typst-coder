#!/usr/bin/env python3
"""Quick test to verify data loading and tokenization."""

import sys
sys.path.insert(0, '.')

from train import load_json_dataset, tokenize_and_group
from transformers import AutoTokenizer

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/qwen3-0.6b", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load small subset
    dataset = load_json_dataset("./data/raw/train.json")
    print(f"Dataset size: {len(dataset)}")

    # Take first 10 examples
    small_dataset = dataset.select(range(min(10, len(dataset))))

    # Test tokenization
    block_size = 512
    separator = "<|im_end|>"

    # Process one batch
    examples = {'text': small_dataset['text']}
    result = tokenize_and_group(examples, tokenizer, block_size, separator)

    print(f"Number of blocks: {len(result['input_ids'])}")
    print(f"Block shape: {len(result['input_ids'][0])}")
    print(f"First block input_ids (first 10 tokens): {result['input_ids'][0][:10]}")

    # Decode back
    decoded = tokenizer.decode(result['input_ids'][0][:50], skip_special_tokens=False)
    print(f"Decoded first 50 tokens:\n{decoded}")

    print("Test passed.")

if __name__ == "__main__":
    main()