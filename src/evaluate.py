"""
Evaluate the fine-tuned Typst-Coder model on the test set.

Reports perplexity and generates sample outputs for qualitative evaluation.
Auto-detects device: CUDA > MPS > CPU.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel

from src.utils import (
    get_device, get_device_str, load_base_model, load_tokenizer,
    format_chat_prompt, clear_cache, LORA_PATH, PROCESSED_DIR,
)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
EVAL_SUBSET = 500  # Max samples to evaluate for perplexity


def evaluate_perplexity(model, tokenizer, dataset, device: torch.device):
    """Calculate perplexity on the evaluation dataset."""
    print(f"\nCalculating perplexity on {len(dataset)} samples ({device.type})...")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            input_ids = torch.tensor([sample["input_ids"]]).to(device)
            labels = torch.tensor([sample["labels"]]).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.shape[1]
            total_tokens += labels.shape[1]

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")

            del input_ids, labels, outputs
            if (i + 1) % 10 == 0:
                clear_cache()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    return perplexity


def generate_samples(model, tokenizer, device: torch.device, num_samples: int = 5):
    """Generate sample outputs using example prompts."""
    print(f"\n{'='*60}")
    print("Generating sample outputs")
    print(f"{'='*60}")

    prompts = [
        "Write a Typst function that calculates the factorial of a number",
        "Create a Typst table with 3 columns: Name, Age, City",
        "Draw a flowchart in Typst showing a login process",
        "Write Typst code to format a thesis title page",
        "Create a Typst macro that converts markdown-style headings to Typst headings",
    ]

    model.eval()

    for i, prompt in enumerate(prompts[:num_samples]):
        print(f"\n{'─'*50}")
        print(f"Prompt: {prompt}")
        print(f"{'─'*50}")

        formatted = format_chat_prompt(prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

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
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        print(response)
        clear_cache()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Typst-Coder")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                        default="auto")
    parser.add_argument("--no-ppl", action="store_true",
                        help="Skip perplexity evaluation")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of generation samples (default: 3)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = load_from_disk(str(PROCESSED_DIR))

    # Load model
    base_model, tokenizer = load_base_model(device=device)
    lora_dir = str(LORA_PATH)

    if os.path.exists(lora_dir):
        print(f"Loading LoRA adapters from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, lora_dir)
    else:
        print(f"No LoRA adapters found at {LORA_PATH}, evaluating base model only.")
        model = base_model

    model.to(device)
    model.eval()

    # Perplexity on a subset
    if not args.no_ppl:
        eval_size = min(EVAL_SUBSET, len(dataset["test"]))
        test_samples = dataset["test"].select(range(eval_size))
        evaluate_perplexity(model, tokenizer, test_samples, device)

    generate_samples(model, tokenizer, device, args.samples)
    clear_cache()


if __name__ == "__main__":
    main()
