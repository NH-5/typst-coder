"""
工具模块 / Utility module

提供常用的工具函数。
/ Provides commonly used utility functions.
"""

import os
from pathlib import Path
from typing import Any
import json


def set_seed(seed: int = 42):
    """
    设置随机种子 / Set random seed

    Args:
        seed: 种子值 / Seed value
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_cuda() -> dict:
    """
    检查CUDA环境 / Check CUDA environment

    Returns:
        CUDA信息字典 / CUDA information dictionary
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def get_memory_usage() -> dict:
    """
    获取内存使用情况 / Get memory usage

    Returns:
        内存信息字典 / Memory information dictionary
    """
    import torch

    info = {}

    if torch.cuda.is_available():
        info["allocated"] = torch.cuda.memory_allocated() / 1e9
        info["reserved"] = torch.cuda.memory_reserved() / 1e9
        info["max_allocated"] = torch.cuda.max_memory_allocated() / 1e9

    return info


def format_bytes(size: int) -> str:
    """
    格式化字节大小 / Format byte size

    Args:
        size: 字节大小 / Byte size

    Returns:
        格式化后的大小字符串 / Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_dir_size(path: Path) -> int:
    """
    获取目录大小 / Get directory size

    Args:
        path: 目录路径 / Directory path

    Returns:
        目录大小（字节）/ Directory size in bytes
    """
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def print_dir_info(path: Path, max_depth: int = 2, indent: int = 0):
    """
    打印目录信息 / Print directory information

    Args:
        path: 目录路径 / Directory path
        max_depth: 最大深度 / Maximum depth
        indent: 缩进 / Indentation
    """
    prefix = "  " * indent

    if path.is_file():
        size = format_bytes(path.stat().st_size)
        print(f"{prefix}{path.name} ({size})")
    else:
        print(f"{prefix}{path.name}/")
        if indent < max_depth:
            for item in sorted(path.iterdir()):
                print_dir_info(item, max_depth, indent + 1)


class ConfigManager:
    """配置管理器 / Configuration manager"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化 / Initialize

        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        self.config_path = Path(config_path)
        self.config = {}

        if self.config_path.exists():
            self.load()

    def load(self):
        """加载配置 / Load configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def save(self):
        """保存配置 / Save configuration"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值 / Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """设置配置值 / Set configuration value"""
        self.config[key] = value
        self.save()


if __name__ == "__main__":
    print("CUDA信息 / CUDA info:", check_cuda())
    print("目录信息 / Directory info:")
    print_dir_info(Path("."))
