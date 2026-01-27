#!/bin/bash
# Training script for Typst Coder LLM

set -e  # Exit on error

# Configuration
MODEL_PATH="./model/qwen3-0.6b"
TRAIN_FILE="./data/raw/train.json"
EVAL_FILE="./data/raw/test.json"
OUTPUT_DIR="./output"
SEED=42

# Training hyperparameters
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=3
LEARNING_RATE=5e-5
WARMUP_STEPS=500
LOGGING_STEPS=100
SAVE_STEPS=500
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=3

# Hardware settings (adjust based on your GPU)
# For RTX 4050 6GB:
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

# Mixed precision (fp16 may cause errors, bf16 if supported)
# Use --fp16 or --bf16 or none
# FP16="--fp16"
# BF16="--bf16"
MIXED_PRECISION="--bf16"  # Change to --fp16 if bf16 not supported

# LoRA settings (uncomment to use LoRA)
USE_LORA="--use_lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting training of Typst Coder LLM..."
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE (accumulation: $GRADIENT_ACCUMULATION_STEPS)"

# Build command
CMD="python train.py \
    --model_path \"$MODEL_PATH\" \
    --train_file \"$TRAIN_FILE\" \
    --eval_file \"$EVAL_FILE\" \
    --output_dir \"$OUTPUT_DIR\" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --seed $SEED"

# Add mixed precision
if [ -n "$MIXED_PRECISION" ]; then
    CMD="$CMD $MIXED_PRECISION"
fi

# Add LoRA if enabled
if [ -n "$USE_LORA" ]; then
    CMD="$CMD $USE_LORA --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

echo "Training completed! Model saved to $OUTPUT_DIR"