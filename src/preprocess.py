"""
Preprocess raw Typst JSON data into tokenized instruction-format datasets.

Raw data: JSON array of {repo, file, language, license, content}
Output: Tokenized HuggingFace Arrow datasets with instruction format.
"""

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PROJECT_PATH = Path(__file__).parent.parent
RAW_TRAIN = PROJECT_PATH / "data/raw/train/typst_train.json"
RAW_TEST = PROJECT_PATH / "data/raw/test/typst_test.json"
PROCESSED_DIR = PROJECT_PATH / "data/processed"
MODEL_PATH = PROJECT_PATH / "model/qwen3.5-0.8b-Base"

MAX_LENGTH = 2048
MIN_CONTENT_CHARS = 30
MAX_CONTENT_CHARS = 8192

SYSTEM_PROMPT = (
    "You are an expert Typst programmer. "
    "Write clean, correct Typst code based on the user's request. "
    "Respond with only the Typst code, no explanations."
)


def _clean_content(content: str) -> str:
    return content.strip()


def _extract_file_path(raw_url: str) -> str:
    """Extract a clean relative file path from a GitHub raw URL.

    e.g. 'https://raw.githubusercontent.com/user/repo/main/src/lib.typ'
      -> 'src/lib.typ'
    """
    if "raw.githubusercontent.com" in raw_url:
        parts = raw_url.split("/")
        # Skip: https:, empty, raw.githubusercontent.com, user, repo, branch
        file_idx = 6  # index after user/repo/branch
        if len(parts) > file_idx:
            return "/".join(parts[file_idx:])
    return raw_url


def _format_instruction(sample: dict) -> str:
    """Format a single sample into Qwen chat format."""
    repo_name = sample["repo"].split("/")[-1] if sample["repo"] else "unknown"
    file_path = _extract_file_path(sample.get("file", "unknown"))
    content = _clean_content(sample.get("content", ""))

    user_prompt = f"Write Typst code for: {file_path}\nRepository: {repo_name}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{content}<|im_end|>"
    )


def load_and_filter(path: Path) -> list[dict]:
    """Load raw JSON, filter invalid samples."""
    with open(path) as f:
        data = json.load(f)

    filtered = []
    for item in data:
        content = item.get("content", "").strip()
        if len(content) < MIN_CONTENT_CHARS:
            continue
        if len(content) > MAX_CONTENT_CHARS:
            continue
        if item.get("language") != "typst":
            continue
        item["content"] = content
        filtered.append(item)

    print(f"Loaded {len(filtered)} samples from {path.name} "
          f"(filtered out {len(data) - len(filtered)})")
    return filtered


def tokenize_function(tokenizer, examples: dict) -> dict:
    """Tokenize formatted text with labels for causal LM."""
    texts = [_format_instruction(s) for s in _dicts_to_samples(examples)]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def _dicts_to_samples(examples: dict) -> list[dict]:
    """Convert batched dict to list of individual sample dicts."""
    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])
    return [
        {k: examples[k][i] for k in keys} for i in range(batch_size)
    ]


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"

    print("Loading training data...")
    train_samples = load_and_filter(RAW_TRAIN)
    train_dataset = Dataset.from_list(train_samples)

    print("Loading test data...")
    test_samples = load_and_filter(RAW_TEST)
    test_dataset = Dataset.from_list(test_samples)

    print("Tokenizing training data...")
    tokenize_fn = lambda x: tokenize_function(tokenizer, x)
    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    print("Tokenizing test data...")
    test_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test",
    )

    dataset = DatasetDict({"train": train_tokenized, "test": test_tokenized})

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dataset.save_to_disk(str(PROCESSED_DIR))
    print(f"Saved processed dataset to {PROCESSED_DIR}")
    print(f"Train: {len(train_tokenized)}, Test: {len(test_tokenized)}")


if __name__ == "__main__":
    main()
