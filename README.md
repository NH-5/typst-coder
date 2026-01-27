# Typst Coder LLM

基于 Qwen3-0.6B 微调的 Typst 代码生成语言模型。

## 项目概述

本项目旨在创建一个专门用于 Typst 排版语言的代码生成模型。通过在海量 Typst 代码数据上微调 Qwen3-0.6B 模型，使模型能够理解 Typst 语法并生成高质量的 Typst 代码。

### 功能
- Typst 代码自动补全
- 从自然语言描述生成 Typst 代码
- Typst 代码错误检查和建议
- 代码片段生成

## 项目结构

```
.
├── data/raw/              # 原始数据集
│   ├── train.json        # 训练数据 (21,069 个样本)
│   └── test.json         # 测试数据 (1,000 个样本)
├── model/qwen3-0.6b/     # 基础模型 (Qwen3-0.6B)
├── train.py              # 训练脚本
├── run_training.sh       # 训练启动脚本
├── requirements.txt      # Python 依赖
├── PLAN.md              # 项目计划文档
└── README.md            # 本文件
```

## 环境配置

### 1. 创建 Conda 环境
```bash
conda create -n typst-coder python=3.14
conda activate typst-coder
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证环境
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 数据准备

数据集已预处理为 JSON 格式，每个样本包含：
```json
{
  "repo": "GitHub 仓库 URL",
  "file": "文件 URL",
  "language": "typst",
  "license": "许可证",
  "content": "Typst 源代码"
}
```

数据集统计：
- 训练集: 21,069 个有效样本
- 测试集: 1,000 个有效样本

## 模型训练

### 快速测试
```bash
# 使用少量数据进行测试
python train.py --output_dir ./test-output --debug_samples 100 --max_steps 5 --max_seq_length 512
```

### 完整训练

#### 选项1：全参数微调
```bash
./run_training.sh
```

#### 选项2：LoRA 微调 (节省显存)
编辑 `run_training.sh`，取消注释 LoRA 相关设置：
```bash
# 在 run_training.sh 中启用：
USE_LORA="--use_lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
```

然后运行：
```bash
./run_training.sh
```

#### 选项3：自定义训练
```bash
python train.py \
  --model_path ./model/qwen3-0.6b \
  --train_file ./data/raw/train.json \
  --eval_file ./data/raw/test.json \
  --output_dir ./output \
  --max_seq_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --fp16  # 或 --bf16，如果 GPU 支持
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_seq_length` | 2048 | 训练序列长度 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 2 | 每设备批量大小 |
| `--gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `--learning_rate` | 5e-5 | 学习率 |
| `--warmup_steps` | 500 | 热身步数 |
| `--fp16` / `--bf16` | 无 | 混合精度训练 |
| `--use_lora` | 无 | 启用 LoRA 微调 |
| `--lora_r` | 8 | LoRA 秩 |
| `--lora_alpha` | 16 | LoRA alpha 参数 |
| `--lora_dropout` | 0.1 | LoRA dropout |

## 模型使用

训练完成后，模型保存在 `output/` 目录中。可以使用 Transformers 库加载：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 生成示例
prompt = "#set page(width: 8.5in, height: 11in)\n#let heading"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

## 性能优化建议

### GPU 显存管理
- RTX 4050 6GB：使用默认参数
- 如果显存不足：
  - 减小 `max_seq_length` (如 1024)
  - 减小 `per_device_train_batch_size` (如 1)
  - 增加 `gradient_accumulation_steps` 保持有效批量大小
  - 启用 `--use_lora`
  - 启用梯度检查点 (`--gradient_checkpointing`，需修改代码)

### 训练速度
- 使用混合精度 (`--fp16` 或 `--bf16`)
- 增加 `preprocessing_num_workers` 加速数据预处理
- 使用更长的序列长度可能提高效率

## 常见问题

### 1. FP16 训练错误
错误信息：`ValueError: Attempting to unscale FP16 gradients.`
解决方案：
- 禁用 `--fp16`，使用纯 FP32 训练
- 或使用 `--bf16` (如果 GPU 支持)
- 或移除 `--fp16` 并让模型使用默认精度

### 2. 显存不足
- 启用 LoRA：`--use_lora`
- 减小批量大小和序列长度
- 启用梯度检查点 (需修改代码)

### 3. 训练不稳定
- 减小学习率
- 增加热身步数
- 使用梯度裁剪 (默认已启用)

### 4. 数据加载问题
- 确保 JSON 文件格式正确
- 检查 `content` 字段是否非空
- 使用 `--debug_samples` 测试数据加载

## 评估指标

训练过程中会记录以下指标：
- 训练损失 (train_loss)
- 评估损失 (eval_loss)
- 学习率变化
- 梯度范数

在测试集上的困惑度 (perplexity) 可作为模型性能的主要指标。

## 后续计划

1. **模型评估**：开发 Typst 代码生成评估基准
2. **指令微调**：收集指令-代码对数据，进行指令微调
3. **编辑器集成**：开发 VSCode 插件
4. **模型量化**：量化模型以便在 CPU 上运行

## 许可证

本项目基于 Apache 2.0 许可证。数据集中的 Typst 代码遵循各自的原始许可证。

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) 团队提供基础模型
- 所有 Typst 开源项目贡献者提供训练数据
- Hugging Face 生态系统提供训练框架

## 联系方式

如有问题或建议，请提交 GitHub Issue 或联系项目维护者。