"""
Typst Coder 源码包

本模块提供Typst代码生成模型的完整功能:
- 数据处理: 数据集下载和预处理
- 模型管理: 模型下载和加载
- 训练: 模型微调训练
- 推理: 代码生成
"""

from . import config
from . import data
from . import model
from . import training
from . import inference
from . import utils

__version__ = "1.0.0"
