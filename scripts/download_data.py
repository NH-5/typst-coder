#!/usr/bin/env python3
"""
下载数据集脚本 / Download dataset script

用法 / Usage:
    python scripts/download_data.py
    python scripts/download_data.py --url <数据集URL> --output <输出路径>
"""

import sys
from pathlib import Path

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DatasetDownloader, DatasetProcessor
from src.config import DATASET_URLS


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="下载 Typst 数据集 / Download Typst dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="原始数据存储目录 / Raw data storage directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="处理后数据存储目录 / Processed data storage directory")
    parser.add_argument("--skip-process", action="store_true",
                        help="跳过数据处理 / Skip data processing")
    args = parser.parse_args()

    print("=" * 50)
    print("Typst 数据集下载工具 / Typst Dataset Download Tool")
    print("=" * 50)

    # 下载数据 / Download data
    downloader = DatasetDownloader(args.raw_dir, args.processed_dir)
    downloader.download_all(DATASET_URLS)

    if not args.skip_process:
        # 处理数据 / Process data
        print("\n开始处理数据... / Starting data processing...")
        processor = DatasetProcessor(args.raw_dir, args.processed_dir)
        processor.process_all()

        # 显示数据信息 / Display data info
        info = processor.get_dataset_info()
        print("\n数据集信息: / Dataset info:")
        for split, data_info in info.items():
            print(f"  {split}: {data_info['samples']} 样本 / samples -> {data_info['path']}")

    print("\n数据准备完成! / Data preparation complete!")


if __name__ == "__main__":
    main()
