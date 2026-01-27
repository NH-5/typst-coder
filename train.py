#!/usr/bin/env python3
"""
Fine-tune Qwen3-0.6B on Typst code dataset.
Uses causal language modeling (CLM) for continued pretraining.
"""

import os
import json
import logging
from typing import Dict, List
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import Dataset
import numpy as np

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_json_dataset(data_path: str) -> Dataset:
    """Load JSON dataset where each line is a JSON object with a 'content' field."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        # The file is a JSON array
        items = json.load(f)
        for item in items:
            content = item.get('content', '')
            if content.strip():
                data.append({'text': content})
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return Dataset.from_list(data)

def tokenize_and_group(examples: Dict[str, List], tokenizer, block_size: int, separator: str) -> Dict[str, List]:
    """Tokenize texts, add separator, and group into blocks of block_size."""
    # Add separator to each text
    texts = [text + separator for text in examples['text']]
    # Tokenize each text individually
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=block_size,  # Truncate long documents
        padding=False,
        return_attention_mask=True,
        return_overflowing_tokens=False,
    )

    # Concatenate all tokenized texts
    concatenated = {k: [] for k in tokenized.keys()}
    for i in range(len(tokenized['input_ids'])):
        for k in tokenized.keys():
            concatenated[k].extend(tokenized[k][i])

    # Group into blocks of block_size
    total_length = len(concatenated['input_ids'])
    if total_length >= block_size:
        # Drop the small remainder
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
    else:
        # If total length is less than block_size, pad or discard?
        # For now, pad to block_size
        result = {
            k: [t + [tokenizer.pad_token_id] * (block_size - len(t)) if k == 'input_ids' else t + [0] * (block_size - len(t))]
            for k, t in concatenated.items()
        }
    return result

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B on Typst code")

    # Model/data paths
    parser.add_argument("--model_path", type=str, default="./model/qwen3-0.6b", help="Path to pretrained model")
    parser.add_argument("--train_file", type=str, default="./data/raw/train.json", help="Training data JSON file")
    parser.add_argument("--eval_file", type=str, default="./data/raw/test.json", help="Evaluation data JSON file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model and logs")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Sequence length for training")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Limit total checkpoints")

    # Other settings
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--document_separator", type=str, default="<|im_end|>", help="Token to separate documents")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of workers for preprocessing")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached datasets")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (overrides num_train_epochs)")
    parser.add_argument("--debug_samples", type=int, default=-1, help="Limit number of samples for debugging")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=False,
    )
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_json_dataset(args.train_file)
    eval_dataset = load_json_dataset(args.eval_file)

    # Debug: limit samples if specified
    if args.debug_samples > 0:
        logger.info(f"DEBUG: Limiting to {args.debug_samples} samples")
        train_dataset = train_dataset.select(range(min(args.debug_samples, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(args.debug_samples, len(eval_dataset))))

    # Process datasets
    logger.info(f"Tokenizing and grouping with block size {args.max_seq_length}...")

    def process_dataset(dataset):
        return dataset.map(
            lambda examples: tokenize_and_group(
                examples, tokenizer, args.max_seq_length, args.document_separator
            ),
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing and grouping",
        )

    train_processed = process_dataset(train_dataset)
    eval_processed = process_dataset(eval_dataset)

    logger.info(f"Train dataset: {len(train_processed)} blocks")
    logger.info(f"Eval dataset: {len(eval_processed)} blocks")

    # Load model
    logger.info("Loading model...")
    # Determine dtype for model
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        trust_remote_code=False,
    )
    model.config.use_cache = False  # Disable cache for training

    # Apply LoRA if requested
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.error("peft library not installed. Install with: pip install peft")
            raise

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        eval_strategy="steps",
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,  # Can enable if memory issues
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=eval_processed,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate
    logger.info("Evaluating...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"Training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()