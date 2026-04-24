"""
Device-agnostic utilities for Typst-Coder training and inference.

Handles device detection (CUDA > MPS > CPU), model loading with optional
QLoRA quantization, and memory management across backends.
"""

import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_PATH = Path(__file__).parent.parent
MODEL_PATH = PROJECT_PATH / "model" / "qwen3.5-0.8b-Base"
LORA_PATH = PROJECT_PATH / "output" / "lora-adapters"
PROCESSED_DIR = PROJECT_PATH / "data" / "processed"

# LoRA defaults
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Tokenizer settings
PAD_TOKEN = "<|endoftext|>"
EOS_TOKEN = "<|im_end|>"

SYSTEM_PROMPT = (
    "You are an expert Typst programmer. "
    "Write clean, correct Typst code based on the user's request. "
    "Respond with only the Typst code, no explanations."
)


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_str() -> str:
    """Return device name as string for logging."""
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16() -> bool:
    """Check if bf16 mixed precision is supported."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def supports_qlora() -> bool:
    """Check if 4-bit quantization is available (CUDA-only via bitsandbytes)."""
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb
        return True
    except ImportError:
        return False


def clear_cache():
    """Clear GPU cache for the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_amp_context():
    """Return the appropriate autocast context for the current device."""
    if torch.cuda.is_available() and supports_bf16():
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    if torch.cuda.is_available():
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    if torch.backends.mps.is_available():
        return lambda: torch.autocast("mps", dtype=torch.float16)
    return lambda: torch.no_grad()


def load_tokenizer():
    """Load tokenizer with correct special tokens for chat format."""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    return tokenizer


def load_base_model(device: torch.device | None = None, use_qlora: bool = False):
    """Load the base Qwen3.5-0.8B model, optionally quantized.

    Args:
        device: Target device. Auto-detected if None.
        use_qlora: Use 4-bit NF4 quantization (CUDA only).

    Returns:
        model, tokenizer
    """
    if device is None:
        device = get_device()

    tokenizer = load_tokenizer()

    model_kwargs = {
        "trust_remote_code": True,
    }

    if use_qlora and supports_qlora():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("Using QLoRA (4-bit NF4 quantization)")
    elif device.type == "cuda":
        model_kwargs["dtype"] = torch.bfloat16 if supports_bf16() else torch.float16
    else:
        model_kwargs["dtype"] = torch.float16

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), **model_kwargs)

    if not use_qlora:
        model.to(device)

    return model, tokenizer


def get_lora_config() -> LoraConfig:
    """Return the standard LoRA configuration."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def format_chat_prompt(user_message: str) -> str:
    """Format a user message into Qwen chat format for inference."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
