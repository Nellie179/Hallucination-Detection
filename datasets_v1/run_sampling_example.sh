#!/bin/bash
# 使用示例：为已有数据生成多次采样

# ==========================================
# 配置区域
# ==========================================

# 输入文件（已有的 metadata）
INPUT_FILE="./experiments/Qwen/Qwen3-8B/coqa_5000samples/03_final_scored_metadata.jsonl"

# 输出文件（包含 stochastic_samples 的新文件）
OUTPUT_FILE="./experiments/Qwen/Qwen3-8B/coqa_5000samples/03_with_stochastic_samples.jsonl"

# 模型配置
MODEL_NAME="Qwen/Qwen3-8B"

# 采样配置
NUM_SAMPLES=10        # 每个样本采样 10 次
TEMPERATURE=0.8       # 采样温度
TOP_P=0.9             # Nucleus sampling 参数

# 其他选项
MAX_NEW_TOKENS=128    # 最大生成 token 数（留空则自动推断）

# ==========================================
# 执行采样
# ==========================================

echo "=================================================="
echo "🎲 开始生成多次采样文本"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "模型: $MODEL_NAME"
echo "采样配置: $NUM_SAMPLES 次 × temperature=$TEMPERATURE"
echo "=================================================="
echo ""

python generate_stochastic_samples.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_NAME" \
    --num-samples $NUM_SAMPLES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-new-tokens $MAX_NEW_TOKENS \
    --trust-remote-code \
    --resume  # 支持断点续传

echo ""
echo "=================================================="
echo "✅ 完成！"
echo "=================================================="
