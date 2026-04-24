"""
LoRA fine-tuning script for Typst-Coder on Apple Silicon (MPS).

Uses LoRA with PEFT on Qwen3.5-0.8B-Base. Optimized for M1 MacBook Air 16GB:
- float16 precision, no quantization (bitsandbytes is CUDA-only)
- Small batch size with gradient accumulation
- Gradient checkpointing for memory efficiency

Usage:
    python -m src.train                  # Full training
    python -m src.train --subset 2000    # Quick test on 2000 samples
    python -m src.train --resume         # Resume from latest checkpoint
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

PROJECT_PATH = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_PATH / "data/processed"
MODEL_PATH = PROJECT_PATH / "model/qwen3.5-0.8b-Base"
OUTPUT_DIR = PROJECT_PATH / "output"
LORA_OUTPUT = OUTPUT_DIR / "lora-adapters"

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500
MAX_GRAD_NORM = 1.0


def load_model_and_tokenizer():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to("mps")

    return model, tokenizer


def apply_lora(model) -> PeftModel:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def main():
    parser = argparse.ArgumentParser(description="Train Typst-Coder with LoRA")
    parser.add_argument("--subset", type=int, default=0,
                        help="Train on a subset of N samples (0 = full dataset)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. This script requires Apple Silicon.")

    print(f"Using device: mps")
    print(f"PyTorch version: {torch.__version__}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    print("Loading processed dataset...")
    dataset = load_from_disk(str(PROCESSED_DIR))

    if args.subset > 0:
        train_size = min(args.subset, len(dataset["train"]))
        eval_size = min(args.subset // 10, len(dataset["test"]))
        dataset["train"] = dataset["train"].select(range(train_size))
        dataset["test"] = dataset["test"].select(range(eval_size))

    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    steps_per_epoch = len(dataset["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print(f"Steps per epoch: ~{steps_per_epoch}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        logging_dir=str(OUTPUT_DIR / "logs"),
        remove_unused_columns=True,
        label_names=["labels"],
        resume_from_checkpoint=args.resume,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"Saving LoRA adapters to {LORA_OUTPUT}...")
    model.save_pretrained(str(LORA_OUTPUT))
    tokenizer.save_pretrained(str(LORA_OUTPUT))

    print("Training complete!")


if __name__ == "__main__":
    main()
