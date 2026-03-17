#!/usr/bin/env python3
"""快速测试批处理功能"""
import os
import time
from hidden_state import HiddenStateExtractor
from adaptive_config import get_adaptive_config

# 使用少量样本快速测试
TEST_INPUT = "./TEST_BATCH/test_10samples.jsonl"
TEST_H5 = "./TEST_BATCH/test_batch.h5"
TEST_JSONL = "./TEST_BATCH/test_batch.jsonl"

os.makedirs("./TEST_BATCH", exist_ok=True)

# 删除旧文件
for f in [TEST_H5, TEST_JSONL]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("🧪 批处理功能测试")
print("=" * 60)

# 获取自适应配置
adaptive_config = get_adaptive_config(
    model_name="Qwen/Qwen3-8B",
    user_model_kwargs={},
    user_generation_kwargs={},
    user_template_kwargs={},
    verbose=False
)

print("\n[*] 加载模型...")
extractor = HiddenStateExtractor(
    model_name="Qwen/Qwen3-8B",
    model_kwargs=adaptive_config["model_kwargs"]
)

print("\n[*] 开始批处理推理 (batch_size=4)...")
start_time = time.time()

extractor.process_from_file(
    input_jsonl_path=TEST_INPUT,
    output_h5_path=TEST_H5,
    output_jsonl_path=TEST_JSONL,
    layer_config={"mode": "middle", "count": 5},
    token_config={"mode": "backward", "count": 5},
    max_new_tokens=256,  # 测试新的 max_new_tokens
    max_queue_size=20,
    system_prompt="You are a helpful, accurate, and honest AI assistant.",
    num_shots=4,
    generation_kwargs=adaptive_config["generation_kwargs"],
    template_kwargs=adaptive_config["template_kwargs"],
    batch_size=4  # 🚀 批处理
)

elapsed = time.time() - start_time

print(f"\n✅ 完成！总耗时: {elapsed:.2f} 秒")
print(f"平均速度: {10 / elapsed:.2f} 样本/秒")
print(f"单样本耗时: {elapsed / 10:.2f} 秒")

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
