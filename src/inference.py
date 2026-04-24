"""
Interactive inference REPL for Typst-Coder.

Loads the LoRA fine-tuned model and generates Typst code from user prompts.
"""

import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_PATH = Path(__file__).parent.parent
MODEL_PATH = PROJECT_PATH / "model/qwen3.5-0.8b-Base"
LORA_PATH = PROJECT_PATH / "output/lora-adapters"

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

SYSTEM_PROMPT = (
    "You are an expert Typst programmer. "
    "Write clean, correct Typst code based on the user's request. "
    "Respond with only the Typst code, no explanations."
)


def load_model():
    """Load base model + LoRA adapters for inference."""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        trust_remote_code=True,
    )

    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA adapters from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, str(LORA_PATH))

    model.to("mps")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
    """Generate Typst code from a user prompt."""
    formatted = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant part
    if "assistant" in full_response:
        full_response = full_response.split("assistant")[-1].strip()

    return full_response


def repl(model, tokenizer):
    """Interactive read-eval-print loop."""
    print("\n" + "=" * 60)
    print("  Typst-Coder — Interactive Inference")
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
            result = generate(model, tokenizer, prompt)
            print(result)
            print()
            torch.mps.empty_cache()
        except Exception as e:
            print(f"Error: {e}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not torch.backends.mps.is_available():
        print("Warning: MPS not available. Using CPU (will be slow).")

    model, tokenizer = load_model()
    repl(model, tokenizer)


if __name__ == "__main__":
    main()
