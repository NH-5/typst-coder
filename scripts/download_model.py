#!/usr/bin/env python3
"""
下载模型脚本 / Download model script

用法 / Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model <模型名称> --output <输出路径>
"""

import sys
from pathlib import Path

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import ModelDownloader


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="下载预训练模型 / Download pretrained model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="HuggingFace模型名称 / HuggingFace model name")
    parser.add_argument("--output", type=str, default="model/qwen3-0.6b",
                        help="本地存储路径 / Local storage path")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace访问令牌 / HuggingFace access token")
    args = parser.parse_args()

    print("=" * 50)
    print("模型下载工具 / Model Download Tool")
    print("=" * 50)
    print(f"模型 / Model: {args.model}")
    print(f"输出 / Output: {args.output}")

    # 下载模型 / Download model
    downloader = ModelDownloader(args.output)
    model_path = downloader.download_model(args.model, args.token)

    print(f"\n模型已下载至 / Model downloaded to: {model_path}")

    # 测试加载分词器 / Test loading tokenizer
    print("\n测试加载分词器... / Testing tokenizer loading...")
    tokenizer = downloader.load_tokenizer()
    print(f"分词器词表大小 / Tokenizer vocabulary size: {len(tokenizer)}")

    print("\n模型下载完成! / Model download complete!")


if __name__ == "__main__":
    main()
