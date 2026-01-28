"""
模型下载和加载模块 / Model download and loading module

提供预训练模型的下载和管理功能。
/ Provides download and management functionality for pretrained models.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelDownloader:
    """模型下载器 / Model downloader"""

    def __init__(self, local_path: str = "model/qwen3-0.6b"):
        """
        初始化下载器 / Initialize downloader

        Args:
            local_path: 本地模型存储路径 / Local model storage path
        """
        self.local_path = Path(local_path)
        self.local_path.mkdir(parents=True, exist_ok=True)

    def download_model(self, model_name: str = "Qwen/Qwen3-0.6B",
                       token: Optional[str] = None,
                       local_path: Optional[str] = None) -> Path:
        """
        下载模型 / Download model

        Args:
            model_name: HuggingFace模型名称 / HuggingFace model name
            token: HuggingFace访问令牌 / HuggingFace access token
            local_path: 自定义本地路径 / Custom local path

        Returns:
            模型本地路径 / Local model path
        """
        target_path = Path(local_path) if local_path else self.local_path

        # 检查模型是否已存在 / Check if model already exists
        if self._is_model_complete(target_path):
            print(f"模型已存在 / Model exists: {target_path}")
            return target_path

        print(f"正在下载模型 / Downloading model: {model_name}")

        # 下载模型文件 / Download model files
        snapshot_download(
            repo_id=model_name,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True
        )

        print(f"模型下载完成 / Model download complete: {target_path}")
        return target_path

    def _is_model_complete(self, path: Path) -> bool:
        """
        检查模型是否完整 / Check if model is complete

        Args:
            path: 模型路径 / Model path

        Returns:
            模型是否完整 / Whether model is complete
        """
        required_files = ["config.json", "tokenizer.json"]
        return all((path / f).exists() for f in required_files)

    def load_tokenizer(self, model_path: Optional[str] = None) -> AutoTokenizer:
        """
        加载分词器 / Load tokenizer

        Args:
            model_path: 模型路径，为None则使用默认路径 / Model path, use default if None

        Returns:
            分词器实例 / Tokenizer instance
        """
        path = model_path or str(self.local_path)
        print(f"加载分词器 / Loading tokenizer: {path}")
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def load_model(self, model_path: Optional[str] = None,
                   device: str = "auto",
                   torch_dtype: str = "auto") -> AutoModelForCausalLM:
        """
        加载模型 / Load model

        Args:
            model_path: 模型路径，为None则使用默认路径 / Model path, use default if None
            device: 设备 ("cuda", "cpu", "auto") / Device ("cuda", "cpu", "auto")
            torch_dtype: torch数据类型 ("auto", "bf16", "fp16") / Torch data type ("auto", "bf16", "fp16")

        Returns:
            模型实例 / Model instance
        """
        path = model_path or str(self.local_path)
        print(f"加载模型 / Loading model: {path}")
        return AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )


class ModelManager:
    """模型管理器 / Model manager"""

    def __init__(self, model_dir: str = "model"):
        """
        初始化管理器 / Initialize manager

        Args:
            model_dir: 模型根目录 / Model root directory
        """
        self.model_dir = Path(model_dir)

    def list_models(self) -> list:
        """
        列出所有本地模型 / List all local models

        Returns:
            模型列表 / Model list
        """
        models = []
        if self.model_dir.exists():
            for item in self.model_dir.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    models.append(item.name)
        return models

    def get_model_info(self, model_name: str) -> dict:
        """
        获取模型信息 / Get model information

        Args:
            model_name: 模型名称 / Model name

        Returns:
            模型信息字典 / Model information dictionary
        """
        model_path = self.model_dir / model_name
        if not model_path.exists():
            return {"error": "模型不存在 / Model not found"}

        info = {"name": model_name, "path": str(model_path)}

        # 检查关键文件 / Check key files
        key_files = ["config.json", "tokenizer.json", "model.safetensors"]
        info["files"] = {f: (model_path / f).exists() for f in key_files}

        return info


def download_qwen3_0_6b(local_path: str = "model/qwen3-0.6b") -> Path:
    """
    下载Qwen3-0.6B模型 / Download Qwen3-0.6B model

    Args:
        local_path: 本地存储路径 / Local storage path

    Returns:
        模型路径 / Model path
    """
    downloader = ModelDownloader(local_path)
    return downloader.download_model("Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    # 下载模型 / Download model
    model_path = download_qwen3_0_6b()
    print(f"模型路径 / Model path: {model_path}")

    # 测试加载 / Test loading
    downloader = ModelDownloader(str(model_path))
    tokenizer = downloader.load_tokenizer()
    print("分词器加载成功 / Tokenizer loaded successfully")
