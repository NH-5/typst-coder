#!/usr/bin/env python3
"""
推理脚本 / Inference script

用法 / Usage:
    # 交互模式 / Interactive mode
    python scripts/infer.py --demo

    # 单次生成 / Single generation
    python scripts/infer.py --instruction "编写一个Hello World程序"

    # 批量生成 / Batch generation
    python scripts/infer.py --batch prompts.json
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import TypstGenerator


def main():
    """主函数 / Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Typst 代码生成 / Typst Code Generator")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径 / Model path")
    parser.add_argument("--demo", action="store_true",
                        help="启动交互模式 / Start interactive mode")
    parser.add_argument("--instruction", type=str, default=None,
                        help="生成指令 / Generation instruction")
    parser.add_argument("--input", type=str, default=None,
                        help="输入文本 / Input text")
    parser.add_argument("--batch", type=str, default=None,
                        help="批量生成文件路径 / Batch generation file path")
    parser.add_argument("--output", type=str, default="output.txt",
                        help="输出文件路径 / Output file path")
    args = parser.parse_args()

    print("=" * 50)
    print("Typst 代码生成器 / Typst Code Generator")
    print("=" * 50)

    if args.demo:
        # 交互模式 / Interactive mode
        from src.inference import interactive_demo
        interactive_demo(args.model)

    elif args.instruction:
        # 单次生成 / Single generation
        generator = TypstGenerator(args.model).load()
        result = generator.generate(args.instruction, args.input, do_stream=False)

        print("\n生成结果 / Generated result:")
        print("-" * 30)
        print(result)

        # 保存结果 / Save result
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n结果已保存至 / Result saved to: {args.output}")

    elif args.batch:
        # 批量生成 / Batch generation
        with open(args.batch, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        generator = TypstGenerator(args.model).load()
        results = generator.batch_generate(prompts)

        # 保存结果 / Save results
        output = []
        for prompt, result in zip(prompts, results):
            output.append({
                "instruction": prompt["instruction"],
                "output": result
            })

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"批量生成完成，结果保存至 / Batch generation complete, result saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
