#!/usr/bin/env python3
"""测试 torch.compile() + Flash Attention 配置"""
import os
import time
from hidden_state import HiddenStateExtractor
from adaptive_config import get_adaptive_config

# 使用少量样本测试
TEST_INPUT = "./TEST_BATCH/test_10samples.jsonl"
TEST_H5 = "./TEST_BATCH/test_compile_fa.h5"
TEST_JSONL = "./TEST_BATCH/test_compile_fa.jsonl"

os.makedirs("./TEST_BATCH", exist_ok=True)

# 删除旧文件
for f in [TEST_H5, TEST_JSONL]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 70)
print("🧪 测试 torch.compile() + Flash Attention")
print("=" * 70)

# 获取自适应配置
adaptive_config = get_adaptive_config(
    model_name="Qwen/Qwen3-8B",
    user_model_kwargs={},
    user_generation_kwargs={},
    user_template_kwargs={},
    verbose=True  # 显示详细信息
)

print("\n[*] 正在加载模型并启用 torch.compile()...")
start_load = time.time()

extractor = HiddenStateExtractor(
    model_name="Qwen/Qwen3-8B",
    model_kwargs=adaptive_config["model_kwargs"],
    use_compile=True  # 🚀 启用 torch.compile()
)

load_time = time.time() - start_load
print(f"[+] 模型加载耗时: {load_time:.2f} 秒")

print("\n[*] 开始批处理推理 (batch_size=16)...")
start_time = time.time()

extractor.process_from_file(
    input_jsonl_path=TEST_INPUT,
    output_h5_path=TEST_H5,
    output_jsonl_path=TEST_JSONL,
    layer_config={"mode": "middle", "count": 5},
    token_config={"mode": "backward", "count": 5},
    max_new_tokens=256,
    max_queue_size=20,
    system_prompt="You are a helpful, accurate, and honest AI assistant.",
    num_shots=4,
    generation_kwargs=adaptive_config["generation_kwargs"],
    template_kwargs=adaptive_config["template_kwargs"],
    batch_size=16  # 🚀 批处理
)

elapsed = time.time() - start_time

print(f"\n{'='*70}")
print(f"✅ 测试完成！")
print(f"{'='*70}")
print(f"模型加载: {load_time:.2f} 秒")
print(f"推理耗时: {elapsed:.2f} 秒")
print(f"总耗时: {load_time + elapsed:.2f} 秒")
print(f"平均速度: {10 / elapsed:.2f} 样本/秒")
print(f"单样本耗时: {elapsed / 10:.2f} 秒")
print(f"{'='*70}")

# 检查生成的 token 数量
import json
print("\n📊 生成 token 数量统计:")
with open(TEST_JSONL, 'r') as f:
    tokens_list = []
    for line in f:
        item = json.loads(line)
        tokens_list.append(item['total_generated_tokens'])
    print(f"  平均: {sum(tokens_list)/len(tokens_list):.1f} tokens")
    print(f"  最大: {max(tokens_list)} tokens")
    print(f"  最小: {min(tokens_list)} tokens")
