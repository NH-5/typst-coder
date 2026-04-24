"""
Evaluate the fine-tuned Typst-Coder model on the test set.

Reports perplexity and generates sample outputs for qualitative evaluation.
"""

import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_PATH = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_PATH / "data/processed"
MODEL_PATH = PROJECT_PATH / "model/qwen3.5-0.8b-Base"
LORA_PATH = PROJECT_PATH / "output/lora-adapters"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7


def load_peft_model():
    """Load base model + LoRA adapters for evaluation."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, str(LORA_PATH))
    model.to("mps")
    model.eval()

    return model, tokenizer


def evaluate_perplexity(model, tokenizer, dataset, split: str = "test"):
    """Calculate perplexity on the evaluation dataset."""
    print(f"\nCalculating perplexity on {split} set...")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            input_ids = torch.tensor([sample["input_ids"]]).to("mps")
            labels = torch.tensor([sample["labels"]]).to("mps")

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.shape[1]
            total_tokens += labels.shape[1]

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")

            del input_ids, labels, outputs
            torch.mps.empty_cache()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    return perplexity


def generate_samples(model, tokenizer, num_samples: int = 5):
    """Generate sample outputs using example prompts."""
    print(f"\nGenerating {num_samples} sample outputs...\n")

    prompts = [
        "Write a Typst function that calculates the factorial of a number",
        "Create a Typst table with 3 columns: Name, Age, City",
        "Draw a flowchart in Typst showing a login process",
        "Write Typst code to format a thesis title page",
        "Create a Typst macro that converts markdown-style headings to Typst headings",
    ]

    for i, prompt in enumerate(prompts[:num_samples]):
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        formatted = (
            f"<|im_start|>system\n"
            f"You are an expert Typst programmer. Write clean, correct Typst code."
            f"<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(formatted, return_tensors="pt").to("mps")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(response)
        torch.mps.empty_cache()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading dataset and model...")
    dataset = load_from_disk(str(PROCESSED_DIR))

    if os.path.exists(LORA_PATH):
        model, tokenizer = load_peft_model()
    else:
        print(f"LoRA adapters not found at {LORA_PATH}")
        print("Loading base model only for baseline evaluation...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.float16,
            trust_remote_code=True,
        )
        model.to("mps")
        model.eval()

    # Evaluate perplexity on a subset for speed
    test_samples = dataset["test"].select(range(min(500, len(dataset["test"]))))
    evaluate_perplexity(model, tokenizer, test_samples)

    generate_samples(model, tokenizer)


if __name__ == "__main__":
    main()
