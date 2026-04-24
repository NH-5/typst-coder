"""
LoRA/QLoRA fine-tuning for Typst-Coder.

Auto-detects device and configures training accordingly:
- CUDA: QLoRA (4-bit) + bf16 mixed precision + flash-attention
- MPS:  LoRA + float16 (no quantization, bitsandbytes is CUDA-only)
- CPU:  LoRA + float32 (fallback)

Usage:
    python -m src.train                          # Full training, auto-detect
    python -m src.train --device cuda --qlora    # Force CUDA with QLoRA
    python -m src.train --device mps             # Force MPS
    python -m src.train --subset 2000            # Quick test on 2000 samples
    python -m src.train --resume                 # Resume from latest checkpoint
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils import (
    get_device, get_device_str, supports_bf16, supports_qlora,
    load_base_model, load_tokenizer, get_lora_config, clear_cache,
    PROCESSED_DIR,
)

PROJECT_PATH = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_PATH / "output"


def build_training_args(args, device) -> TrainingArguments:
    """Build TrainingArguments with device-appropriate defaults."""
    use_bf16 = supports_bf16() and not args.no_bf16
    use_cuda = device.type == "cuda"

    # CUDA can use mixed precision via fp16/bf16 flags; MPS needs manual amp
    fp16 = use_cuda and not use_bf16
    bf16 = use_cuda and use_bf16

    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        optim="adamw_torch" if use_cuda else "adamw_torch",
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        eval_strategy="steps" if args.save_steps > 0 else "epoch",
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=2 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
        report_to="none",
        logging_dir=str(OUTPUT_DIR / "logs"),
        remove_unused_columns=True,
        resume_from_checkpoint=args.resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Typst-Coder with LoRA/QLoRA")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                        default="auto", help="Target device (default: auto-detect)")
    parser.add_argument("--qlora", action="store_true",
                        help="Use 4-bit QLoRA quantization (CUDA only)")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 even if supported")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio (default: 0.03)")
    parser.add_argument("--lr-scheduler", default="cosine",
                        choices=["cosine", "linear", "constant"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--subset", type=int, default=0,
                        help="Train on subset of N samples (0 = full)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    # --- Device setup ---
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if args.device != "auto" and args.device != get_device().type:
        print(f"Warning: forcing device={args.device} but {get_device_str()} is available")

    use_qlora = args.qlora and supports_qlora()
    if args.qlora and not supports_qlora():
        print("Warning: QLoRA requested but bitsandbytes not available. Falling back to LoRA.")

    print(f"Device: {get_device_str()}")
    print(f"QLoRA: {use_qlora}, bf16: {supports_bf16() and not args.no_bf16}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Model ---
    model, tokenizer = load_base_model(device=device, use_qlora=use_qlora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = get_lora_config()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # --- Data ---
    print("Loading processed dataset...")
    dataset = load_from_disk(str(PROCESSED_DIR))

    if args.subset > 0:
        train_size = min(args.subset, len(dataset["train"]))
        eval_size = min(args.subset // 10, len(dataset["test"]))
        dataset["train"] = dataset["train"].select(range(train_size))
        dataset["test"] = dataset["test"].select(range(eval_size))

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(dataset["train"]) // eff_batch
    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    print(f"Effective batch size: {eff_batch}, Steps/epoch: ~{steps_per_epoch}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = build_training_args(args, device)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    lora_out = OUTPUT_DIR / "lora-adapters"
    print(f"Saving LoRA adapters to {lora_out}...")
    peft_model.save_pretrained(str(lora_out))
    tokenizer.save_pretrained(str(lora_out))

    clear_cache()
    print("Training complete!")


if __name__ == "__main__":
    main()
