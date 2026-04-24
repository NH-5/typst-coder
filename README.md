# Typst-Coder

基于 Qwen3.5-0.8B-Base 微调的 Typst 代码生成模型。

## 项目结构

```
typst-coder/
├── src/
│   ├── preprocess.py   # 数据预处理：格式化、分词、保存
│   ├── train.py         # LoRA 微调训练
│   ├── evaluate.py      # 模型评估（困惑度 + 生成样例）
│   └── inference.py     # 交互式推理 REPL
├── downloader/
│   ├── data_download.py # 下载 HuggingFace 数据集
│   └── model_download.py# 下载基座模型
├── data/
│   ├── raw/             # 原始 JSON 数据
│   └── processed/       # 分词后的 Arrow 数据集
├── model/
│   └── qwen3.5-0.8b-Base/  # 基座模型权重
└── output/
    └── lora-adapters/   # 训练后的 LoRA 权重
```

## 环境要求

- Python >= 3.13
- Apple Silicon Mac (MPS)，16GB+ 内存
- 依赖见 `pyproject.toml`

## 使用方法

### 1. 下载数据和模型

```bash
python downloader/model_download.py
python downloader/data_download.py
```

### 2. 数据预处理

```bash
python -m src.preprocess
```

将原始 JSON 格式化为 Qwen chat 格式，分词后保存到 `data/processed/`。

### 3. 训练

```bash
python -m src.train
```

使用 LoRA 在 MPS 上进行微调。训练配置：
- LoRA rank=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
- batch_size=1, gradient_accumulation_steps=8
- learning_rate=2e-4, cosine schedule
- 3 epochs, max_length=2048

### 4. 评估

```bash
python -m src.evaluate
```

计算测试集困惑度并生成样例输出。

### 5. 推理

```bash
python -m src.inference
```

启动交互式 REPL，输入需求描述即可生成 Typst 代码。

## 技术方案

- **LoRA 微调**：仅训练低秩适配器，参数量约 0.3% 的可训练参数
- **MPS 后端**：在 Apple Silicon 上进行 float16 训练，无需 CUDA
- **梯度检查点**：节省训练时显存
- **数据格式**：Qwen chat 格式（system/user/assistant）
