"""
训练模块 / Training module

提供Typst代码生成模型的训练功能，支持LoRA微调。
/ Provides training functionality for Typst code generation model, supports LoRA fine-tuning.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import json

# 添加src目录到路径 / Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
from ..config import TRAINING_CONFIG


class TypstDataCollator(DataCollatorForSeq2Seq):
    """Typst数据集收集器 / Typst dataset collator"""

    def __init__(self, tokenizer, padding=True):
        super().__init__(tokenizer, padding=padding)


class TypstTrainer:
    """Typst代码生成模型训练器 / Typst code generation model trainer"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化训练器 / Initialize trainer

        Args:
            config: 训练配置，为None则使用默认配置 / Training configuration, use default if None
        """
        self.config = config or TRAINING_CONFIG
        self.tokenizer = None
        self.model = None

    def prepare_tokenizer(self) -> AutoTokenizer:
        """
        准备分词器 / Prepare tokenizer

        Returns:
            分词器实例 / Tokenizer instance
        """
        print("加载分词器... / Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name_or_path"],
            trust_remote_code=True,
            padding_side="right"
        )

        # 设置特殊token / Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def prepare_model(self):
        """
        准备模型 / Prepare model
        """
        print("加载模型... / Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name_or_path"],
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )

        # 如果启用LoRA，应用LoRA配置 / If LoRA is enabled, apply LoRA configuration
        if self.config.get("use_lora", False):
            self._apply_lora()

        print(f"模型参数量 / Model parameters: {self.model.num_parameters() / 1e6:.2f}M")

    def _apply_lora(self):
        """
        应用LoRA微调 / Apply LoRA fine-tuning
        """
        print("应用LoRA配置... / Applying LoRA configuration...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=self.config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self, split: str = "train") -> Dataset:
        """
        加载数据集 / Load dataset

        Args:
            split: 数据集分割 ("train" 或 "test") / Dataset split ("train" or "test")

        Returns:
            HuggingFace数据集 / HuggingFace dataset
        """
        data_path = Path(self.config["data_dir"]) / f"{split}.json"

        if not data_path.exists():
            raise FileNotFoundError(f"数据集不存在 / Dataset not found: {data_path}")

        print(f"加载{split}数据集 / Loading {split} dataset: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为Dataset格式 / Convert to Dataset format
        dataset = Dataset.from_list(data)

        # 过滤空样本 / Filter empty samples
        dataset = dataset.filter(lambda x: x.get("text", "").strip() != "")

        return dataset

    def tokenize_function(self, examples):
        """
        Tokenize样本 / Tokenize samples

        Args:
            examples: 样本字典 / Samples dictionary

        Returns:
            Tokenized样本 / Tokenized samples
        """
        max_length = self.config.get("max_seq_length", 1024)

        # Tokenize文本 / Tokenize text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # 设置labels（与input_ids相同，用于语言模型训练）
        # Set labels (same as input_ids, for language model training)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(self) -> tuple:
        """
        准备训练和评估数据集 / Prepare training and evaluation datasets

        Returns:
            (训练数据集, 评估数据集) / (training_dataset, evaluation_dataset)
        """
        train_dataset = self.load_dataset("train")
        eval_dataset = self.load_dataset("test")

        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text"]
        )

        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text"]
        )

        print(f"训练集样本数 / Training samples: {len(train_dataset)}")
        print(f"评估集样本数 / Evaluation samples: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def create_training_args(self) -> TrainingArguments:
        """
        创建训练参数 / Create training arguments

        Returns:
            训练参数实例 / Training arguments instance
        """
        output_dir = self.config["output_dir"]

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_steps=self.config.get("warmup_steps", 500),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            fp16=self.config.get("fp16", True),
            ddp_find_unused_parameters=self.config.get("ddp_find_unused_parameters", False),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_steps=self.config.get("eval_steps", 500),
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config.get("save_total_limit", 3),
            do_train=self.config.get("do_train", True),
            do_eval=self.config.get("do_eval", True),
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

    def train(self):
        """
        执行训练 / Execute training
        """
        print("=" * 50)
        print("开始训练 Typst 代码生成模型 / Start training Typst code generation model")
        print("=" * 50)

        # 准备分词器和模型 / Prepare tokenizer and model
        self.prepare_tokenizer()
        self.prepare_model()

        # 准备数据集 / Prepare dataset
        train_dataset, eval_dataset = self.prepare_dataset()

        # 创建数据收集器 / Create data collator
        data_collator = TypstDataCollator(
            tokenizer=self.tokenizer,
            padding=True
        )

        # 创建训练参数 / Create training arguments
        training_args = self.create_training_args()

        # 创建训练器 / Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # 开始训练 / Start training
        print("\n开始训练... / Starting training...")
        trainer.train()

        # 保存模型 / Save model
        print("\n保存模型... / Saving model...")
        trainer.save_model(training_args.output_dir)
        self.tokenizer.save_pretrained(training_args.output_dir)

        # 保存训练配置 / Save training configuration
        with open(Path(training_args.output_dir) / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print(f"\n训练完成! 模型保存至 / Training complete! Model saved to: {training_args.output_dir}")

        return trainer


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="训练 Typst 代码生成模型 / Train Typst code generation model")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 / Configuration file path")
    args = parser.parse_args()

    # 如果指定了配置文件，则加载 / Load configuration if specified
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        trainer = TypstTrainer(config)
    else:
        trainer = TypstTrainer()

    trainer.train()


if __name__ == "__main__":
    main()
