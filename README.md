# typst-coder

`typst-coder` 是一个面向 Typst 代码建模的小型实验项目，目标是基于 `Qwen3.5-0.8B-Base` 训练一个更懂 Typst 语法、结构与常见用法的代码模型。

当前仓库主要包含三部分内容：

- 数据下载脚本
- 数据清洗脚本
- 训练路线与方案文档

目前仓库还没有正式的训练脚本、评测脚本或推理接口。

## 项目目标

项目当前采用两阶段路线：

1. Continued pretraining
2. Task SFT

第一阶段的目标是让模型先学会 Typst 分布本身，包括：

- Typst 语法与常见结构
- 模板、宏、package 的使用方式
- 真实 Typst 仓库中的文档组织模式

第二阶段再把这些领域知识转成更具体的任务能力，包括：

- 续写 / 补全
- 中间补全
- 结构化编辑
- 报错修复

更完整的路线说明见 [docs/roadmap.md](./docs/roadmap.md)。

## 当前状态

当前已经具备：

- 从 Hugging Face 下载公开 Typst 数据集
- 将原始数据按 `typst` / `markdown` 分离
- 进行基础长度过滤与去重
- 下载基础模型到本地目录
- 用文档明确训练路线、数据策略和阶段目标

当前尚未实现：

- continued pretraining 训练代码
- repo-level 数据划分脚本
- task SFT 样本构造脚本
- 自动化评测与 compile benchmark
- 推理服务或 CLI

## 环境要求

- Python `>= 3.13`
- 推荐使用 `uv` 管理环境

## 快速开始

### 1. 安装依赖

### 2. 下载数据集

```bash
uv run python data_download.py
```

该脚本会从 Hugging Face 下载：

- `TechxGenus/Typst-Train`
- `TechxGenus/Typst-Test`

默认输出目录：

```text
data/raw/train/typst_train.json
data/raw/test/typst_test.json
```

### 3. 清洗数据

```bash
uv run python data_clean.py
```

该脚本当前会：

- 删除缺失值
- 删除 `license` 与 `file` 列
- 按 `language` 将数据分为 `typst` 与 `markdown`
- 过滤短文本
  - 少于 `80` 个字符
  - 少于 `8` 行
- 去除完全重复行
- 对 `repo` 做一次去重

默认输出目录：

```text
data/cleaned/train/typst.json
data/cleaned/train/no_typst.json
data/cleaned/test/typst.json
data/cleaned/test/no_typst.json
```

### 4. 下载基础模型

```bash
uv run python model_download.py
```

当前脚本会下载：

- `Qwen/Qwen3.5-0.8B-Base`

默认输出目录：

```text
model/qwen3.6-0.8b/
```

说明：目录名当前写成了 `qwen3.6-0.8b`，但下载的模型仓库是 `Qwen3.5-0.8B-Base`。

## 数据流程

当前项目的数据流很直接：

```text
Hugging Face Dataset
        |
        v
data/raw/train/typst_train.json
data/raw/test/typst_test.json
        |
        v
data_clean.py
        |
        v
data/cleaned/{train,test}/typst.json
data/cleaned/{train,test}/no_typst.json
```

现阶段仓库里的清洗逻辑是“可运行的第一版”，而更完整的数据清洗规则定义在 [docs/data-cleaning.md](./docs/data-cleaning.md)。两者并不完全一致。

## 目录结构

```text
typst-coder/
├── data/
│   ├── raw/
│   └── cleaned/
├── docs/
│   ├── roadmap.md
│   ├── data-cleaning.md
│   ├── continued-pretraining.md
│   └── task-sft.md
├── model/
├── data_download.py
├── data_clean.py
├── model_download.py
├── pyproject.toml
└── README.md
```

## 核心脚本

- `data_download.py`
        下载训练集与测试集到本地 `data/raw/`。
- `data_clean.py`
        对原始 JSON 数据做基础清洗，并分别导出 `typst` 与 `markdown` 子集。
- `model_download.py`
        下载基座模型到本地 `model/`。

## 文档导航

- [docs/roadmap.md](./docs/roadmap.md)：项目整体路线总览
- [docs/data-cleaning.md](./docs/data-cleaning.md)：第一阶段数据清洗规则
- [docs/continued-pretraining.md](./docs/continued-pretraining.md)：continued pretraining 方案
- [docs/task-sft.md](./docs/task-sft.md)：task SFT 方案

## 已知限制

- 当前清洗逻辑仍然比较基础，和文档中定义的完整规则还有差距
- 还没有训练、验证、评测和推理部分的实现

## License

项目使用 [MIT License](./LICENSE)。
