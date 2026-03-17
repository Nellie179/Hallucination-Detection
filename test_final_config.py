#!/usr/bin/env python3
"""
完整测试当前配置：
- SDPA attention
- torch.compile()
- batch_size=16
- left padding
- max_new_tokens=2048
"""
import os
import time
import json
from hidden_state import HiddenStateExtractor
from adaptive_config import get_adaptive_config

print("=" * 80)
print("🧪 完整配置测试")
print("=" * 80)

# 使用 10 个样本测试
TEST_INPUT = "./TEST_BATCH/test_10samples.jsonl"
TEST_H5 = "./TEST_BATCH/final_test.h5"
TEST_JSONL = "./TEST_BATCH/final_test.jsonl"

os.makedirs("./TEST_BATCH", exist_ok=True)

# 清理旧文件
for f in [TEST_H5, TEST_JSONL]:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 80)
print("第 1 步：检测 GPU 和配置")
print("=" * 80)

adaptive_config = get_adaptive_config(
    model_name="Qwen/Qwen3-8B",
    user_model_kwargs={},
    user_generation_kwargs={},
    user_template_kwargs={},
    verbose=True
)

# 检查关键配置
print("\n✓ 关键配置检查:")
print(f"  - Attention 实现: {adaptive_config['model_kwargs'].get('attn_implementation', 'unknown')}")
print(f"  - 数据类型: {adaptive_config['model_kwargs'].get('dtype', 'unknown')}")
print(f"  - Thinking 模式: {adaptive_config['template_kwargs'].get('enable_thinking', False)}")

print("\n" + "=" * 80)
print("第 2 步：加载模型（启用 torch.compile）")
print("=" * 80)

start_load = time.time()
extractor = HiddenStateExtractor(
    model_name="Qwen/Qwen3-8B",
    model_kwargs=adaptive_config["model_kwargs"],
    use_compile=True  # 🚀 启用编译
)
load_time = time.time() - start_load

print(f"\n✓ 模型加载完成: {load_time:.2f} 秒")

# 检查 tokenizer padding
print(f"✓ Tokenizer padding side: {extractor.tokenizer.padding_side}")
if extractor.tokenizer.padding_side != 'left':
    print("  ⚠️  警告: padding_side 应该是 'left'")

print("\n" + "=" * 80)
print("第 3 步：批处理推理测试 (10 样本, batch_size=16)")
print("=" * 80)

start_time = time.time()

extractor.process_from_file(
    input_jsonl_path=TEST_INPUT,
    output_h5_path=TEST_H5,
    output_jsonl_path=TEST_JSONL,
    layer_config={"mode": "middle", "count": 5},
    token_config={"mode": "backward", "count": 5},
    max_new_tokens=2048,  # 使用真实配置
    max_queue_size=20,
    system_prompt="You are a helpful, accurate, and honest AI assistant.",
    num_shots=4,
    generation_kwargs=adaptive_config["generation_kwargs"],
    template_kwargs=adaptive_config["template_kwargs"],
    batch_size=16
)

elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("第 4 步：性能统计")
print("=" * 80)

print(f"\n⏱️  性能数据:")
print(f"  - 模型加载: {load_time:.2f} 秒")
print(f"  - 推理耗时: {elapsed:.2f} 秒")
print(f"  - 总耗时: {load_time + elapsed:.2f} 秒")
print(f"  - 平均速度: {10 / elapsed:.2f} 样本/秒")
print(f"  - 单样本耗时: {elapsed / 10:.2f} 秒")

# 检查生成的 token 数量
print(f"\n📊 生成 token 统计:")
with open(TEST_JSONL, 'r') as f:
    tokens_list = []
    for line in f:
        item = json.loads(line)
        tokens_list.append(item['total_generated_tokens'])

    print(f"  - 平均: {sum(tokens_list)/len(tokens_list):.1f} tokens")
    print(f"  - 最大: {max(tokens_list)} tokens")
    print(f"  - 最小: {min(tokens_list)} tokens")

    # 检查是否有被截断到 2048 的样本
    truncated = sum(1 for t in tokens_list if t == 2048)
    if truncated > 0:
        print(f"  ⚠️  {truncated} 个样本被截断到 2048 tokens")

print("\n" + "=" * 80)
print("第 5 步：检查警告和错误")
print("=" * 80)

print("\n✓ 检查结果:")
print("  - 如果上面没有 padding_side 警告 → ✓ 批处理配置正确")
print("  - 如果看到 'torch.compile() 启用成功' → ✓ 编译加速已启用")
print("  - 如果看到 'Batch inferencing' → ✓ 批处理正常工作")
print(f"  - 如果 Attention 实现是 SDPA → ✓ 使用高效注意力机制")

print("\n" + "=" * 80)
print("✅ 配置测试完成！")
print("=" * 80)

# 性能对比
print("\n📈 性能对比:")
print(f"  最初版本 (单样本): ~11.3 秒/样本")
print(f"  当前版本 (优化后): {elapsed / 10:.2f} 秒/样本")
if elapsed / 10 < 11.3:
    speedup = 11.3 / (elapsed / 10)
    print(f"  提速倍数: {speedup:.2f}x ({(speedup-1)*100:.0f}% 提升)")

print("\n💡 下一步:")
print("  如果所有检查都通过，可以开始生成大规模数据集！")
print("  修改 generate_pipeline.py 中的 max_samples 参数即可。")
