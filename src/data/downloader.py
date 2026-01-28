"""
数据集下载和预处理模块 / Dataset download and preprocessing module

提供Typst数据集的下载、解析和预处理功能。
/ Provides download, parsing and preprocessing functionality for Typst datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm


class DatasetDownloader:
    """数据集下载器 / Dataset downloader"""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        初始化下载器 / Initialize downloader

        Args:
            raw_dir: 原始数据存储目录 / Raw data storage directory
            processed_dir: 处理后数据存储目录 / Processed data storage directory
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # 确保目录存在 / Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, url: str, filename: str) -> Path:
        """
        下载数据集文件 / Download dataset file

        Args:
            url: 数据集下载链接 / Dataset download URL
            filename: 保存的文件名 / Save filename

        Returns:
            下载文件的路径 / Downloaded file path
        """
        output_path = self.raw_dir / filename

        # 如果文件已存在，跳过下载 / Skip download if file exists
        if output_path.exists():
            print(f"文件已存在 / File exists: {output_path}")
            return output_path

        print(f"正在下载 / Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"下载完成 / Download complete: {output_path}")
        return output_path

    def download_all(self, urls: Dict[str, str]) -> Dict[str, Path]:
        """
        下载多个数据集 / Download multiple datasets

        Args:
            urls: 数据集名称到URL的映射 / Dataset name to URL mapping

        Returns:
            下载文件路径的字典 / Dictionary of downloaded file paths
        """
        results = {}
        for name, url in urls.items():
            filename = f"{name}.json"
            results[name] = self.download_dataset(url, filename)
        return results


class DatasetProcessor:
    """数据集处理器 / Dataset processor"""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        初始化处理器 / Initialize processor

        Args:
            raw_dir: 原始数据目录 / Raw data directory
            processed_dir: 处理后数据目录 / Processed data directory
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_json(self, filepath: Path) -> List[Dict]:
        """
        加载JSON文件 / Load JSON file

        Args:
            filepath: JSON文件路径 / JSON file path

        Returns:
            数据列表 / Data list
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def format_sample(self, sample: Dict) -> Optional[Dict]:
        """
        格式化单个样本 / Format single sample

        Args:
            sample: 原始样本数据 / Raw sample data

        Returns:
            格式化后的样本，非typst数据返回None / Formatted sample, returns None for non-typst data
        """
        # 只处理Typst代码 / Only process Typst code
        if sample.get("language") != "typst":
            return None

        if "content" in sample:
            # 生成任务: 根据代码文件生成文档或测试
            # Generation task: generate documentation or tests from code file
            instruction = "请为以下Typst代码编写文档说明，包括功能描述和使用示例。"
            input_text = sample.get("content", "")
            output_text = "# Documentation\n\n(此处应填写代码文档)"

            text = f"### Instruction:\n{instruction}\n\n"
            text += f"### Input:\n{input_text}\n\n"
            text += f"### Output:\n{output_text}"
            return {"text": text}
        elif "instruction" in sample and "output" in sample:
            # 格式化训练样本 / Format training sample
            text = f"### Instruction:\n{sample['instruction']}\n\n"
            if "input" in sample and sample["input"]:
                text += f"### Input:\n{sample['input']}\n\n"
            text += f"### Output:\n{sample['output']}"
            return {"text": text}
        else:
            return None

    def process_dataset(self, split: str = "train") -> Path:
        """
        处理数据集 / Process dataset

        Args:
            split: 数据集分割 ("train" 或 "test") / Dataset split ("train" or "test")

        Returns:
            处理后文件的路径 / Processed file path
        """
        input_path = self.raw_dir / f"{split}.json"
        output_path = self.processed_dir / f"{split}.json"

        if not input_path.exists():
            raise FileNotFoundError(f"原始数据文件不存在 / Raw data file not found: {input_path}")

        print(f"正在处理 {split} 数据集... / Processing {split} dataset...")

        # 加载原始数据 / Load raw data
        data = self.load_json(input_path)

        # 格式化每个样本 / Format each sample
        formatted = [self.format_sample(sample) for sample in data]

        # 过滤非Typst数据 / Filter non-Typst data
        processed_data = [f for f in formatted if f is not None]

        # 保存处理后的数据 / Save processed data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成 / Processing complete: {output_path} ({len(processed_data)} 样本 / samples)")
        return output_path

    def process_all(self) -> Dict[str, Path]:
        """
        处理所有数据集 / Process all datasets

        Returns:
            处理后文件路径的字典 / Dictionary of processed file paths
        """
        results = {}
        for split in ["train", "test"]:
            try:
                results[split] = self.process_dataset(split)
            except FileNotFoundError:
                print(f"警告 / Warning: {split} 数据集不存在 / dataset not found")
        return results

    def get_dataset_info(self) -> Dict:
        """
        获取数据集信息 / Get dataset information

        Returns:
            数据集信息字典 / Dataset information dictionary
        """
        info = {}
        for split in ["train", "test"]:
            input_path = self.raw_dir / f"{split}.json"
            output_path = self.processed_dir / f"{split}.json"

            if output_path.exists():
                data = self.load_json(output_path)
                info[split] = {
                    "samples": len(data),
                    "path": str(output_path)
                }
        return info


def download_and_process(urls: Dict[str, str], raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> Dict:
    """
    一键下载并处理数据集 / Download and process dataset in one step

    Args:
        urls: 数据集URL字典 / Dataset URL dictionary
        raw_dir: 原始数据目录 / Raw data directory
        processed_dir: 处理后数据目录 / Processed data directory

    Returns:
        处理后文件路径的字典 / Dictionary of processed file paths
    """
    # 下载数据 / Download data
    downloader = DatasetDownloader(raw_dir, processed_dir)
    downloader.download_all(urls)

    # 处理数据 / Process data
    processor = DatasetProcessor(raw_dir, processed_dir)
    return processor.process_all()


if __name__ == "__main__":
    from config import DATASET_URLS

    results = download_and_process(DATASET_URLS)
    print("\n数据集信息 / Dataset info:")
    for name, path in results.items():
        print(f"  {name}: {path}")
