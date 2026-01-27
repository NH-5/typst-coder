#!/bin/bash
# Server training script for Typst Coder LLM
# This script sets up environment, downloads resources, and runs training on a server

set -e  # Exit on error
set -x  # Print commands for debugging

# Configuration - EDIT THESE VARIABLES BEFORE RUNNING
# ======================================================

# 1. Dataset URLs (REQUIRED)
# Provide URLs to download training and test datasets
# Example: TRAIN_URL="https://your-storage.com/train.json"
#          TEST_URL="https://your-storage.com/test.json"
TRAIN_URL="https://huggingface.co/datasets/TechxGenus/Typst-Train/resolve/main/typst_train.json?download=true"  # REQUIRED: URL to training data
TEST_URL="https://huggingface.co/datasets/TechxGenus/Typst-Test/resolve/main/typst_test.json?download=true"   # REQUIRED: URL to test data

# 2. Model settings
MODEL_ID="Qwen/Qwen3-0.6B"  # Hugging Face model ID
MODEL_REVISION="main"        # Model revision/branch

# 3. Training settings
OUTPUT_DIR="./output"        # Where to save trained model
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=3
LEARNING_RATE=5e-5
BATCH_SIZE=2                 # Per device batch size
GRAD_ACCUMULATION=8          # Gradient accumulation steps
USE_LORA=true                # Use LoRA for efficient fine-tuning
MIXED_PRECISION="bf16"       # "fp16", "bf16", or "" for none

# 4. Environment settings
CONDA_ENV_NAME="typst-coder"  # Conda environment name
PYTHON_VERSION="3.14"         # Python version

# 5. Directory structure
BASE_DIR="$(pwd)"
MODEL_DIR="${BASE_DIR}/model"
DATA_DIR="${BASE_DIR}/data/raw"

# ======================================================
# END OF CONFIGURATION
# ======================================================

# Validate configuration
validate_config() {
    if [ -z "$TRAIN_URL" ]; then
        echo "ERROR: TRAIN_URL is not set. Please edit this script and provide the URL to training data."
        exit 1
    fi

    if [ -z "$TEST_URL" ]; then
        echo "ERROR: TEST_URL is not set. Please edit this script and provide the URL to test data."
        exit 1
    fi

    echo "Configuration validated:"
    echo "  Train URL: $TRAIN_URL"
    echo "  Test URL: $TEST_URL"
    echo "  Model: $MODEL_ID"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  Conda env: $CONDA_ENV_NAME"
}

# Setup Conda environment
setup_conda() {
    echo "=== Setting up Conda environment ==="

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi

    # Create or activate conda environment
    if conda info --envs | grep -q "^${CONDA_ENV_NAME}[[:space:]]"; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists. Activating..."
        conda activate "${CONDA_ENV_NAME}"
    else
        echo "Creating conda environment '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
        conda create -y -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}"
        conda activate "${CONDA_ENV_NAME}"
    fi

    # Verify activation
    if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
        echo "ERROR: Failed to activate conda environment."
        exit 1
    fi

    echo "Conda environment ready: $(which python)"
}

# Install dependencies
install_dependencies() {
    echo "=== Installing Python dependencies ==="

    pip install --upgrade pip

    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "WARNING: requirements.txt not found. Installing basic dependencies..."
        pip install torch transformers datasets accelerate peft \
                   huggingface-hub requests numpy tqdm scikit-learn evaluate
    fi

    echo "Dependencies installed."
}

# Download resources
download_resources() {
    echo "=== Downloading resources ==="

    # Create directories
    mkdir -p "${MODEL_DIR}"
    mkdir -p "${DATA_DIR}"

    # Run download script
    python download_resources.py \
        --model-id "${MODEL_ID}" \
        --model-dir "${MODEL_DIR}" \
        --model-revision "${MODEL_REVISION}" \
        --train-url "${TRAIN_URL}" \
        --test-url "${TEST_URL}" \
        --data-dir "${DATA_DIR}"

    echo "Resources downloaded."
}

# Run training
run_training() {
    echo "=== Starting training ==="

    # Build training command
    CMD="python train.py \
        --model_path \"${MODEL_DIR}/Qwen3-0.6B\" \
        --train_file \"${DATA_DIR}/train.json\" \
        --eval_file \"${DATA_DIR}/test.json\" \
        --output_dir \"${OUTPUT_DIR}\" \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --warmup_steps 500 \
        --logging_steps 100 \
        --save_steps 500 \
        --eval_steps 500 \
        --save_total_limit 3 \
        --seed 42"

    # Add mixed precision
    if [ -n "${MIXED_PRECISION}" ]; then
        if [ "${MIXED_PRECISION}" = "fp16" ]; then
            CMD="${CMD} --fp16"
        elif [ "${MIXED_PRECISION}" = "bf16" ]; then
            CMD="${CMD} --bf16"
        fi
    fi

    # Add LoRA if enabled
    if [ "${USE_LORA}" = "true" ]; then
        CMD="${CMD} --use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"
    fi

    echo "Running training command:"
    echo "${CMD}"
    echo ""

    # Execute training
    eval ${CMD}

    echo "Training completed!"
}

# Main execution flow
main() {
    echo "========================================="
    echo "Typst Coder LLM - Server Training Script"
    echo "========================================="

    # Step 1: Validate configuration
    validate_config

    # Step 2: Setup conda environment
    setup_conda

    # Step 3: Install dependencies
    install_dependencies

    # Step 4: Download resources
    download_resources

    # Step 5: Run training
    run_training

    echo "========================================="
    echo "All steps completed successfully!"
    echo "Trained model saved to: ${OUTPUT_DIR}"
    echo "========================================="
}

# Run main function
main "$@"