#!/usr/bin/env python3
"""
训练脚本 / Training script

用法 / Usage:
    python scripts/train.py
    python scripts/train.py --config config/training_config.json
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training import TypstTrainer
from src.utils import set_seed, check_cuda


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="训练 Typst 代码生成模型 / Train Typst code generation model")
    parser.add_argument("--config", type=str, default=None,
                        help="训练配置文件路径 / Training configuration file path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数 / Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批次大小 / Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 / Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 / Random seed")
    args = parser.parse_args()

    print("=" * 50)
    print("Typst 代码生成模型训练 / Typst Code Generation Model Training")
    print("=" * 50)

    # 检查环境 / Check environment
    cuda_info = check_cuda()
    print(f"CUDA可用 / CUDA available: {cuda_info['cuda_available']}")
    if cuda_info.get('device_name'):
        print(f"设备 / Device: {cuda_info['device_name']}")
        print(f"显存 / VRAM: {cuda_info.get('device_memory', 0):.2f} GB")

    # 设置随机种子 / Set random seed
    set_seed(args.seed)

    # 加载配置 / Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        from src.config import TRAINING_CONFIG
        config = TRAINING_CONFIG.copy()

    # 命令行参数覆盖配置 / Command line arguments override configuration
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr

    print(f"\n训练配置 / Training config:")
    print(f"  模型 / Model: {config['model_name_or_path']}")
    print(f"  训练轮数 / Epochs: {config['num_train_epochs']}")
    print(f"  批次大小 / Batch size: {config['per_device_train_batch_size']}")
    print(f"  学习率 / Learning rate: {config['learning_rate']}")
    print(f"  输出 / Output: {config['output_dir']}")

    # 创建训练器并训练 / Create trainer and train
    trainer = TypstTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
