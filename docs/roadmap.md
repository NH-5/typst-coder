# Typst Coder 训练路线总览

本文档用于汇总当前 `typst coder` 项目的整体训练路线。目标不是一次性覆盖所有可能方向，而是先固定一个可执行、可验证、可逐步扩展的默认方案。

当前路线分为两个核心阶段：

1. 第一阶段：continued pretraining
2. 第二阶段：task SFT

当前版本暂不进入 RL、preference learning 和大规模自然语言到完整 Typst 文档生成。

## 1. 项目目标

本项目的目标是基于 `Qwen3.5-0.8B-Base`，训练一个面向 Typst 的小型代码模型。

第一阶段先解决：

- 模型是否真正学会 Typst 语法、结构、模板和常见用法

第二阶段再解决：

- 模型是否能够完成 Typst 补全、编辑和修复等具体任务

因此，当前默认路线不是直接追求“全能对话式 Typst 助手”，而是先做一个能力边界清晰、训练闭环完整的 `typst coder`。

## 2. 数据来源

当前使用的数据集为：

- 训练集：`TechxGenus/Typst-Train`
- 测试集：`TechxGenus/Typst-Test`

当前默认做法是：

- 只保留 `language == typst`
- 不在第一阶段引入 Markdown
- 如果后续需要更强的自然语言到 Typst 能力，再单独加入高价值 Typst 相关文档语料

## 3. 第一阶段：continued pretraining

第一阶段目标：

- 让模型进一步学习 Typst 分布
- 强化 Typst 语法、结构、宏和 package 使用能力
- 在 repo-level 验证集上获得更好的语言建模能力

第一阶段默认策略：

- 使用 `Qwen3.5-0.8B-Base`
- 只做标准 `next-token prediction`
- 不做 fill-in-the-middle
- 不做 instruction tuning
- 不做 RL

第一阶段数据策略：

- 使用 `typst-only` 清洗结果
- 做 packing
- 保留显式样本边界
- 超长文件按固定长度切块

第一阶段评测策略：

- 使用 `repo-level` 划分
- 默认比例：`train/dev/test = 85/10/5`
- `Typst-Test` 仅作为辅助 benchmark
- 固定训练预算，不使用 early stopping
- 训练过程中定期评估 checkpoint
- 训练结束后按 `dev_repo loss` 选择最佳 checkpoint

第一阶段建议预算：

- 第一轮默认跑 `3` 个 corpus pass

详细方案见：

- [data-cleaning.md](/Users/wuzheng/projects/typst-coder/docs/data-cleaning.md)
- [continued-pretraining.md](/Users/wuzheng/projects/typst-coder/docs/continued-pretraining.md)

## 4. 第二阶段：task SFT

第二阶段目标：

- 把第一阶段学到的 Typst 知识转化为具体任务能力

当前第二阶段第一版只做四类任务：

1. 续写 / 补全
2. 中间补全
3. 结构化编辑
4. 报错修复

第二阶段当前默认策略：

- 从现有 `typst-only` 语料自动构造任务样本
- 不优先依赖外部 instruction 数据
- 使用统一任务模板
- 默认采用片段输出，而不是整篇重写

四类任务在第一版中的定位：

- `续写 / 补全`：主任务
- `中间补全`：次主任务
- `结构化编辑`：受控编辑能力起点
- `报错修复`：可验证修复能力补充

第二阶段第一轮默认训练配比：

- `续写 / 补全`：`70%`
- `中间补全`：`30%`

跑通基线后，再升级到完整混训版本：

- `续写 / 补全`：`55%`
- `中间补全`：`25%`
- `结构化编辑`：`15%`
- `报错修复`：`5%`

第二阶段评测原则：

- 使用分任务评测面板
- 主指标优先看 `strict compile set` 上的 `compile pass rate`
- 同时看任务成功率和人工抽样
- 不强行压成单一总分

当前 `strict compile set` 的口径为：

- package 允许自由使用和安装
- 缺 package 不视为模型失败
- 缺字体不视为模型失败
- 依赖外部资源的样本不进入 `strict compile set`
- 依赖仓库内其他本地子文件的样本不进入 `strict compile set`

也就是说，当前主 compile 指标应优先建立在“单文件、资源依赖可控、可在统一环境下独立稳定编译”的样本上。

详细方案见：

- [task-sft.md](/Users/wuzheng/projects/typst-coder/docs/task-sft.md)

## 5. 当前明确不做的事情

为保证路线聚焦，当前版本明确不优先做以下内容：

- 第一阶段加入 fill-in-the-middle
- 第一阶段加入 Markdown 混合训练
- 第一阶段使用 early stopping
- 第二阶段默认输出整篇文档
- 第二阶段大规模开放式自然语言到完整 Typst 生成
- preference learning
- RL

这些方向不是永远不做，而是当前版本先不作为默认路线。

## 6. 当前推荐执行顺序

建议按以下顺序实现：

1. 完成 `typst-only` 数据清洗
2. 完成 repo-level 数据划分
3. 跑第一阶段 continued pretraining 基线
4. 选出第一阶段最佳 checkpoint
5. 构造第二阶段 `续写 / 中间补全` 样本并跑 SFT 基线
6. 再逐步加入 `结构化编辑` 和 `报错修复`

## 7. 当前版本的判断标准

如果要判断当前路线是否成功，建议看两个层面：

### 7.1 第一阶段是否成功

- `dev_repo loss` 明显优于基座模型
- `Typst-Test loss` 也有改善
- 人工抽查显示 Typst 风格和结构更稳定

### 7.2 第二阶段是否成功

- `strict compile set` 上的 `compile pass rate` 提升
- 补全和中间补全任务的可用性明显提升
- 编辑和修复任务在引入后没有明显拖垮主任务表现

## 8. 当前文档结构

当前项目文档结构如下：

- [data-cleaning.md](/Users/wuzheng/projects/typst-coder/docs/data-cleaning.md)
- [continued-pretraining.md](/Users/wuzheng/projects/typst-coder/docs/continued-pretraining.md)
- [task-sft.md](/Users/wuzheng/projects/typst-coder/docs/task-sft.md)
- [roadmap.md](/Users/wuzheng/projects/typst-coder/docs/roadmap.md)

后续如果继续细化，可以优先补充：

- task SFT 的指令模板示例
- compile 评测环境定义
- 编辑任务与修复任务的更细粒度验证规则
