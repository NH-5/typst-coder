# Typst Coder LLM 微调计划

## 项目目标
基于 Qwen3-0.6B 模型，在 Typst 代码数据集上微调，得到一个能够生成、补全 Typst 代码的专用语言模型。

## 数据集
- 数据位置：`data/raw/train.json`、`data/raw/test.json`
- 格式：JSON 数组，每个元素包含 `content` 字段（Typst 源代码）
- 训练集：147,484 个样本
- 测试集：7,001 个样本

## 模型
- 基础模型：Qwen3-0.6B（存储在 `model/qwen3-0.6b/`）
- 模型参数：6亿参数
- 最大序列长度：131,072（tokenizer 配置）
- Tokenizer：Qwen2Tokenizer，支持特殊 tokens

## 微调方案
采用**因果语言建模（Causal Language Modeling，CLM）**进行继续预训练（continued pretraining）。

### 数据预处理
1. 提取每个样本的 `content` 字段
2. 在每个文档末尾添加分隔符（默认 `<|im_end|>`）
3. 使用 tokenizer 进行分词
4. 将多个文档的 token 连接起来，然后按固定长度（如 2048）分块
5. 训练时使用标准语言建模损失（预测下一个 token）

### 训练配置
- 序列长度：2048（可调整）
- 批量大小：通过梯度累积调整有效批量大小
- 优化器：AdamW
- 学习率：5e-5
- 训练轮数：3
- 支持 LoRA（低秩适应）以节省显存

### 硬件需求
- GPU：NVIDIA GeForce RTX 4050（6GB 显存）
- 内存：建议 16GB 以上系统内存
- 存储：至少 10GB 空闲空间用于保存模型和检查点

## 实施步骤

### 1. 环境准备
```bash
conda activate typst-coder
pip install -r requirements.txt
```

### 2. 数据检查
- 验证数据集格式
- 统计 token 长度分布
- 检查 tokenizer 对 Typst 代码的处理

### 3. 训练脚本
主脚本：`train.py`

支持参数：
- `--model_path`: 模型路径
- `--train_file`/`--eval_file`: 数据文件
- `--output_dir`: 输出目录
- `--max_seq_length`: 序列长度（默认 2048）
- `--use_lora`: 启用 LoRA
- 其他训练超参数

### 4. 训练执行
```bash
# 全参数微调
python train.py --output_dir ./output --fp16 --max_seq_length 2048

# LoRA 微调
python train.py --output_dir ./output-lora --use_lora --fp16 --max_seq_length 2048
```

### 5. 评估与测试
- 在测试集上计算困惑度（perplexity）
- 人工检查生成样本的质量
- 可能开发一个简单的 Typst 代码补全演示

### 6. 模型使用
微调后的模型可以用于：
- Typst 代码自动补全
- 从自然语言描述生成 Typst 代码
- Typst 代码错误检查

## 注意事项
1. 显存管理：6GB 显存可能限制批量大小，建议使用梯度累积和 LoRA
2. 过拟合：监控训练/验证损失，必要时早停
3. 数据质量：数据集来自 GitHub，可能存在噪声，但整体质量较好
4. 分词效率：Typst 代码包含特殊字符，tokenizer 可能需要处理

## 扩展方向
1. 指令微调：如果获得指令-代码对数据，可进行指令微调
2. 更长上下文：Typst 文档可能较长，可尝试更长序列训练
3. 集成到编辑器：开发 VSCode/IntelliJ 插件

## 文件结构
```
.
├── data/raw/              # 原始数据
├── model/qwen3-0.6b/      # 基础模型
├── train.py              # 训练脚本
├── requirements.txt      # 依赖
├── PLAN.md              # 本计划文件
└── output/              # 训练输出（将创建）
```

## 风险与缓解
- 风险：训练不稳定
  - 缓解：使用较小的学习率，监控损失曲线
- 风险：显存不足
  - 缓解：启用 LoRA，减少批量大小，使用梯度检查点
- 风险：过拟合
  - 缓解：使用验证集早停，数据增强

## 成功标准
- 验证集困惑度低于基础模型
- 人工评估生成代码的语法正确性
- 模型能够完成简单的 Typst 代码补全任务

---

*最后更新：2026-01-27*