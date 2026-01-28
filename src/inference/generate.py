"""
推理模块 / Inference module

提供Typst代码生成模型的推理功能。
/ Provides inference functionality for Typst code generation model.
"""

import sys
from pathlib import Path
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# 添加src目录到路径 / Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from ..config import INFERENCE_CONFIG


class TypstGenerator:
    """Typst代码生成器 / Typst code generator"""

    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[dict] = None):
        """
        初始化生成器 / Initialize generator

        Args:
            model_path: 模型路径 / Model path
            config: 推理配置 / Inference configuration
        """
        self.config = config or INFERENCE_CONFIG
        self.model_path = model_path or self.config["model_path"]
        self.tokenizer = None
        self.model = None

    def load(self):
        """
        加载模型和分词器 / Load model and tokenizer
        """
        print(f"加载模型 / Loading model: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 尝试加载LoRA adapter或完整模型 / Try to load LoRA adapter or full model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )

            # 检查是否有LoRA权重 / Check if LoRA weights exist
            adapter_path = Path(self.model_path) / "adapter_model"
            if adapter_path.exists():
                print("检测到LoRA适配器，加载中... / Detected LoRA adapter, loading...")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.model_path
                )
            else:
                self.model = base_model

        except Exception as e:
            print(f"加载失败 / Load failed: {e}")
            raise

        print("模型加载成功! / Model loaded successfully!")
        return self

    def format_prompt(self, instruction: str, input_text: Optional[str] = None) -> str:
        """
        格式化提示词 / Format prompt

        Args:
            instruction: 指令 / Instruction
            input_text: 输入文本（可选）/ Input text (optional)

        Returns:
            格式化后的提示词 / Formatted prompt
        """
        prompt = f"### Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"\n### Input:\n{input_text}\n"
        prompt += "\n### Output:\n"
        return prompt

    def generate(self, instruction: str,
                 input_text: Optional[str] = None,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 do_stream: bool = True) -> str:
        """
        生成Typst代码 / Generate Typst code

        Args:
            instruction: 指令 / Instruction
            input_text: 输入文本 / Input text
            max_new_tokens: 最大生成token数 / Maximum generated tokens
            temperature: 温度 / Temperature
            top_p: top-p采样参数 / Top-p sampling parameter
            do_stream: 是否流式输出 / Whether to stream output

        Returns:
            生成的文本 / Generated text
        """
        if self.model is None:
            self.load()

        # 使用默认参数或覆盖 / Use default parameters or override
        max_new_tokens = max_new_tokens or self.config.get("max_new_tokens", 512)
        temperature = temperature or self.config.get("temperature", 0.7)
        top_p = top_p or self.config.get("top_p", 0.9)

        # 格式化提示词 / Format prompt
        prompt = self.format_prompt(instruction, input_text)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成参数 / Generation parameters
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_stream:
            # 流式输出 / Streaming output
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    streamer=streamer,
                    **generate_kwargs
                )
            return ""
        else:
            # 非流式输出 / Non-streaming output
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            # 解码输出（去掉输入部分）/ Decode output (remove input part)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return generated_text

    def generate_from_sample(self, instruction: str, input_text: Optional[str] = None) -> str:
        """
        根据样本格式生成代码 / Generate code from sample format

        Args:
            instruction: 指令 / Instruction
            input_text: 输入文本 / Input text

        Returns:
            生成的Typst代码 / Generated Typst code
        """
        return self.generate(instruction, input_text, do_stream=False)

    def batch_generate(self, prompts: List[dict]) -> List[str]:
        """
        批量生成 / Batch generation

        Args:
            prompts: 提示词列表，每个元素包含instruction和可选的input
                    / Prompt list, each element contains instruction and optional input

        Returns:
            生成的文本列表 / List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                instruction=prompt["instruction"],
                input_text=prompt.get("input"),
                do_stream=False
            )
            results.append(result)
        return results


def interactive_demo(model_path: Optional[str] = None):
    """
    交互式演示 / Interactive demo

    Args:
        model_path: 模型路径 / Model path
    """
    generator = TypstGenerator(model_path).load()

    print("\n" + "=" * 50)
    print("Typst 代码生成器 - 交互模式 / Typst Code Generator - Interactive Mode")
    print("输入 'quit' 或 'exit' 退出 / Enter 'quit' or 'exit' to quit")
    print("=" * 50)

    while True:
        print("\n" + "-" * 30)
        instruction = input("Instruction: ").strip()

        if instruction.lower() in ["quit", "exit"]:
            print("再见! / Goodbye!")
            break

        if not instruction:
            continue

        input_text = input("Input (可选 / optional): ").strip()

        print("\n生成结果 / Generated result:")
        generator.generate(instruction, input_text if input_text else None)


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Typst 代码生成器 / Typst Code Generator")
    parser.add_argument("--model", type=str, default=None, help="模型路径 / Model path")
    parser.add_argument("--demo", action="store_true", help="启动交互模式 / Start interactive mode")
    parser.add_argument("--instruction", type=str, default=None, help="输入指令 / Input instruction")
    parser.add_argument("--input", type=str, default=None, help="输入文本 / Input text")
    args = parser.parse_args()

    if args.demo:
        interactive_demo(args.model)
    elif args.instruction:
        generator = TypstGenerator(args.model).load()
        result = generator.generate(args.instruction, args.input, do_stream=False)
        print("\n生成结果 / Generated result:")
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
