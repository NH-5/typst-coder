#!/usr/bin/env python3
"""
Download resources for Typst Coder LLM training on server.
Downloads model from Hugging Face Hub and datasets from configurable URLs.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} to {output_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)

    logger.info(f"Download completed: {output_path}")

def download_model(
    model_id: str = "Qwen/Qwen3-0.6B",
    local_dir: Path = Path("./model"),
    revision: str = "main"
):
    """Download model from Hugging Face Hub."""
    model_dir = local_dir / model_id.split("/")[-1]

    if model_dir.exists():
        logger.info(f"Model directory already exists: {model_dir}")
        # Check if essential files exist
        essential_files = ["config.json", "model.safetensors", "tokenizer.json"]
        missing_files = [f for f in essential_files if not (model_dir / f).exists()]
        if not missing_files:
            logger.info("All essential model files present.")
            return
        else:
            logger.warning(f"Missing model files: {missing_files}")

    logger.info(f"Downloading model {model_id} to {model_dir}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            revision=revision,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[
                "*.json", "*.txt", "*.safetensors", "*.model", "*.bin", "*.py"
            ],
        )
        logger.info(f"Model downloaded successfully to {model_dir}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

def download_dataset(
    train_url: str,
    test_url: str,
    output_dir: Path = Path("./data/raw")
):
    """Download training and test datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"

    # Download training data
    if train_url:
        download_file(train_url, train_path)
    else:
        logger.warning("No training data URL provided.")

    # Download test data
    if test_url:
        download_file(test_url, test_path)
    else:
        logger.warning("No test data URL provided.")

    # Verify downloads
    if train_path.exists():
        logger.info(f"Training data size: {train_path.stat().st_size:,} bytes")
    if test_path.exists():
        logger.info(f"Test data size: {test_path.stat().st_size:,} bytes")

def main():
    parser = argparse.ArgumentParser(description="Download resources for Typst Coder LLM")

    # Model settings
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B",
                       help="Hugging Face model ID (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--model-dir", type=str, default="./model",
                       help="Local directory to save model (default: ./model)")
    parser.add_argument("--model-revision", type=str, default="main",
                       help="Model revision/branch (default: main)")

    # Data settings
    parser.add_argument("--train-url", type=str, required=True,
                       help="URL to download training data (train.json)")
    parser.add_argument("--test-url", type=str, required=True,
                       help="URL to download test data (test.json)")
    parser.add_argument("--data-dir", type=str, default="./data/raw",
                       help="Local directory to save data (default: ./data/raw)")

    # Skip options
    parser.add_argument("--skip-model", action="store_true",
                       help="Skip model download")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data download")

    args = parser.parse_args()

    logger.info("Starting resource download...")

    # Download model
    if not args.skip_model:
        try:
            download_model(
                model_id=args.model_id,
                local_dir=Path(args.model_dir),
                revision=args.model_revision,
            )
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping model download.")

    # Download data
    if not args.skip_data:
        try:
            download_dataset(
                train_url=args.train_url,
                test_url=args.test_url,
                output_dir=Path(args.data_dir),
            )
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping data download.")

    logger.info("Resource download completed successfully.")

    # Verify structure
    model_path = Path(args.model_dir) / args.model_id.split("/")[-1]
    train_path = Path(args.data_dir) / "train.json"
    test_path = Path(args.data_dir) / "test.json"

    logger.info("Verifying downloaded resources:")
    logger.info(f"  Model directory: {model_path} (exists: {model_path.exists()})")
    logger.info(f"  Training data: {train_path} (exists: {train_path.exists()})")
    logger.info(f"  Test data: {test_path} (exists: {test_path.exists()})")

if __name__ == "__main__":
    main()