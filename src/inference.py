"""
Interactive inference REPL for Typst-Coder.

Loads the LoRA fine-tuned model and generates Typst code from user prompts.
Auto-detects device: CUDA > MPS > CPU.
"""

import argparse
import os
import sys
from pathlib import Path

_omp_val = os.environ.get("OMP_NUM_THREADS", "")
if _omp_val in ("", "0"):
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
from peft import PeftModel

from src.utils import (
    get_device, get_device_str, load_base_model, load_tokenizer,
    format_chat_prompt, clear_cache, LORA_PATH,
)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def generate(model, tokenizer, prompt: str, device: torch.device,
             max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE) -> str:
    formatted = format_chat_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in full_response:
        full_response = full_response.split("assistant")[-1].strip()

    return full_response


def repl(model, tokenizer, device: torch.device):
    print("\n" + "=" * 60)
    print(f"  Typst-Coder — Interactive Inference ({device.type})")
    print("  Type 'exit' or 'quit' to stop.")
    print("  Type 'clear' to reset the terminal.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if prompt.lower() == "clear":
            os.system("clear" if sys.platform == "darwin" else "cls")
            continue

        print("\nGenerating...\n")
        try:
            result = generate(model, tokenizer, prompt, device)
            print(result)
            print()
            clear_cache()
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Typst-Coder inference REPL")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                        default="auto")
    parser.add_argument("--prompt", type=str, default="",
                        help="Single-shot generation (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    base_model, tokenizer = load_base_model(device=device)

    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA adapters from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    else:
        print(f"No LoRA adapters at {LORA_PATH}, using base model.")
        model = base_model

    model.to(device)
    model.eval()

    if args.prompt:
        result = generate(model, tokenizer, args.prompt, device,
                          max_tokens=args.max_tokens, temperature=args.temperature)
        print(result)
    else:
        repl(model, tokenizer, device)

    clear_cache()


if __name__ == "__main__":
    main()
