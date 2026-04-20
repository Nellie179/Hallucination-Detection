#!/usr/bin/env python3
"""
快速测试脚本 - 验证多次采样功能
在小数据上快速测试，无需等待长时间
"""

import os
import json
import tempfile

# 创建临时测试数据
def create_test_data():
    """创建 2 个测试样本"""
    test_data = [
        {
            "sample_id": "test_001",
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
            "model_output_text": "Paris",
            "structured_data": {
                "task_type": "qa",
                "question": "What is the capital of France?",
                "ground_truths": ["Paris"]
            }
        },
        {
            "sample_id": "test_002",
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho wrote Romeo and Juliet?<|im_end|>\n<|im_start|>assistant\n",
            "model_output_text": "William Shakespeare",
            "structured_data": {
                "task_type": "qa",
                "question": "Who wrote Romeo and Juliet?",
                "ground_truths": ["William Shakespeare", "Shakespeare"]
            }
        }
    ]
    return test_data


def main():
    print("=" * 70)
    print("🧪 多次采样功能快速测试")
    print("=" * 70)

    # 创建临时目录
    test_dir = "./TEST_SAMPLING"
    os.makedirs(test_dir, exist_ok=True)

    # 写入测试数据
    input_file = os.path.join(test_dir, "test_input.jsonl")
    output_file = os.path.join(test_dir, "test_output.jsonl")

    print(f"\n[1/4] 创建测试数据...")
    test_data = create_test_data()
    with open(input_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"   ✓ 已创建 {len(test_data)} 个测试样本")

    # 构建命令
    print(f"\n[2/4] 执行采样生成...")
    cmd = f"""
python generate_stochastic_samples.py \\
    --input {input_file} \\
    --output {output_file} \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --num-samples 3 \\
    --temperature 0.8 \\
    --max-new-tokens 50 \\
    --trust-remote-code
""".strip()

    print(f"   执行命令:")
    print(f"   {cmd}")
    print()

    # 执行
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   ✗ 执行失败:")
        print(result.stderr)
        return

    print(result.stdout)

    # 验证输出
    print(f"\n[3/4] 验证输出...")
    if not os.path.exists(output_file):
        print(f"   ✗ 输出文件不存在: {output_file}")
        return

    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"   ✓ 输出文件包含 {len(lines)} 行")

        # 检查第一个样本
        if lines:
            first_item = json.loads(lines[0])

            # 检查字段
            has_samples = "stochastic_samples" in first_item
            print(f"   ✓ 包含 stochastic_samples 字段: {has_samples}")

            if has_samples:
                samples = first_item["stochastic_samples"]
                print(f"   ✓ 采样数量: {len(samples)}")
                print(f"\n   样本展示:")
                for i, sample in enumerate(samples, 1):
                    print(f"      {i}. {sample}")

    # 清理（可选）
    print(f"\n[4/4] 测试完成！")
    print(f"   测试文件保留在: {test_dir}/")
    print(f"   如需清理，运行: rm -rf {test_dir}")

    print("\n" + "=" * 70)
    print("✅ 测试通过！可以在真实数据上使用了。")
    print("=" * 70)


if __name__ == "__main__":
    main()
