# 多次采样文本生成器使用指南

## 📋 功能说明

`generate_stochastic_samples.py` 是一个独立的轻量级工具，用于为已有的 metadata 文件追加多次随机采样文本。

**核心特性**：
- ✅ 轻量级：仅生成文本，不提取隐藏状态（节省内存和时间）
- ✅ 独立运行：不修改原有 pipeline
- ✅ 断点续传：支持中断后继续（自动保存进度）
- ✅ 灵活配置：采样次数、温度、top-p 等参数可调
- ✅ 进度显示：使用 tqdm 显示实时进度

---

## 🚀 快速开始

### 方式 1：使用示例脚本

```bash
cd /home/zfang1/Data/Lxy/Benchmark/data

# 编辑 run_sampling_example.sh 修改配置
vim run_sampling_example.sh

# 运行
./run_sampling_example.sh
```

### 方式 2：命令行直接调用

```bash
python generate_stochastic_samples.py \
    --input experiments/Qwen/Qwen3-8B/coqa_5000samples/03_final_scored_metadata.jsonl \
    --output experiments/Qwen/Qwen3-8B/coqa_5000samples/03_with_samples.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 10 \
    --temperature 0.8 \
    --trust-remote-code \
    --resume
```

---

## 📖 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | ✅ | - | 输入 JSONL 文件路径（必须包含 `prompt` 字段） |
| `--output` | ✅ | - | 输出 JSONL 文件路径 |
| `--model` | ✅ | - | 模型名称（如 `Qwen/Qwen3-8B`） |
| `--num-samples` | ❌ | 10 | 每个样本采样次数 |
| `--temperature` | ❌ | 0.8 | 采样温度（0.7-1.0 推荐） |
| `--top-p` | ❌ | 0.9 | Nucleus sampling 参数 |
| `--max-new-tokens` | ❌ | 自动推断 | 最大生成 token 数 |
| `--device` | ❌ | 自动检测 | 设备（`cuda` 或 `cpu`） |
| `--trust-remote-code` | ❌ | False | 信任远程代码（Qwen 等模型需要） |
| `--resume` | ❌ | False | 断点续传模式 |

---

## 💡 使用场景

### 场景 1：为小数据集生成采样（测试）

```bash
python generate_stochastic_samples.py \
    --input experiments/Qwen/Qwen3-8B/coqa_100samples/03_final_scored_metadata.jsonl \
    --output experiments/Qwen/Qwen3-8B/coqa_100samples/03_with_samples.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 5 \
    --temperature 0.7 \
    --trust-remote-code
```

**适用于**：快速验证功能是否正常

---

### 场景 2：为大数据集生成采样（生产）

```bash
python generate_stochastic_samples.py \
    --input experiments/Qwen/Qwen3-8B/coqa_5000samples/03_final_scored_metadata.jsonl \
    --output experiments/Qwen/Qwen3-8B/coqa_5000samples/03_with_samples.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 10 \
    --temperature 0.8 \
    --trust-remote-code \
    --resume  # 重要：启用断点续传
```

**适用于**：生产环境，长时间运行，支持中断恢复

---

### 场景 3：高温度采样（增加多样性）

```bash
python generate_stochastic_samples.py \
    --input experiments/Qwen/Qwen3-8B/coqa_1000samples/03_final_scored_metadata.jsonl \
    --output experiments/Qwen/Qwen3-8B/coqa_1000samples/03_high_temp_samples.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 20 \
    --temperature 1.0 \
    --top-p 0.95 \
    --trust-remote-code
```

**适用于**：Semantic Entropy（需要高多样性）

---

## 🔍 输出格式

输出的 JSONL 文件在原有字段基础上，新增 `stochastic_samples` 字段：

```json
{
    "sample_id": "coqa_train_000000",
    "prompt": "...",
    "model_output_text": "Paris",
    "eval_category": "correct",
    "structured_data": {...},
    "stochastic_samples": [
        "Paris",
        "The capital of France is Paris.",
        "Paris is the capital.",
        "It is Paris.",
        "France's capital is Paris.",
        ...
    ]
}
```

---

## 🛠️ 断点续传机制

### 工作原理

1. 每处理完一个样本，立即保存检查点到 `<output>.checkpoint`
2. 如果程序中断（Ctrl+C 或崩溃），再次运行时加上 `--resume`
3. 程序会跳过已处理的样本，继续未完成的部分

### 使用示例

```bash
# 第一次运行（假设处理了 500/5000）
python generate_stochastic_samples.py \
    --input data.jsonl \
    --output output.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 10 \
    --resume

# 按 Ctrl+C 中断...

# 第二次运行（自动从第 501 个继续）
python generate_stochastic_samples.py \
    --input data.jsonl \
    --output output.jsonl \
    --model Qwen/Qwen3-8B \
    --num-samples 10 \
    --resume  # 必须保留此参数
```

**注意**：
- 检查点文件：`output.jsonl.checkpoint`
- 全部完成后自动删除检查点文件
- 如需重新开始，删除检查点文件或去掉 `--resume` 参数

---

## ⚡ 性能优化建议

### 1. 内存优化

如果显存不足，可以降低模型精度或使用 CPU：

```bash
# 使用 float16（降低显存占用）
# 修改代码中的 dtype=torch.float16

# 或使用 CPU（速度慢但不占显存）
python generate_stochastic_samples.py \
    --device cpu \
    ...
```

### 2. 速度优化

- **减少采样次数**：先用 `--num-samples 5` 测试
- **使用更快的模型**：如 Qwen-1.5B 而非 Qwen-8B
- **批量处理**（未实现）：修改代码支持 batch generation

### 3. 采样质量 vs 效率

| 配置 | 质量 | 多样性 | 速度 | 适用场景 |
|------|------|--------|------|----------|
| `temp=0.7, num=5` | 中 | 低 | 快 | 测试 |
| `temp=0.8, num=10` | 高 | 中 | 中 | 生产（推荐） |
| `temp=1.0, num=20` | 高 | 高 | 慢 | Semantic Entropy |

---

## 🔗 与 Baseline Detectors 集成

生成的文件可以直接用于 baseline detectors：

### 1. 修改配置

```python
# baseline_detectors/config.py

# 使用包含采样的文件
METADATA_JSONL = os.path.join(
    EXPERIMENT_DIR,
    "03_with_stochastic_samples.jsonl"  # ✨ 改为新文件
)
```

### 2. 验证数据

```python
# 快速验证
import json
with open("path/to/03_with_stochastic_samples.jsonl", 'r') as f:
    sample = json.loads(f.readline())
    print("Has stochastic_samples:", "stochastic_samples" in sample)
    print("Number of samples:", len(sample.get("stochastic_samples", [])))
```

### 3. 使用 SampleAccessor

```python
# baseline_detectors 中自动可用
accessor = SampleAccessor(sample_id, metadata, h5_group)
samples = accessor.get_stochastic_samples()  # 自动读取
print(f"Got {len(samples)} samples")
```

---

## 📊 预计耗时

基于 Qwen-8B 在 A100 上的测试：

| 数据集大小 | 采样次数 | 预计耗时 | 输出文件大小 |
|-----------|---------|---------|-------------|
| 100 样本 | 5 次 | ~5 分钟 | ~500 KB |
| 1000 样本 | 10 次 | ~1 小时 | ~5 MB |
| 5000 样本 | 10 次 | ~5 小时 | ~25 MB |

**注意**：实际耗时取决于：
- 硬件配置（GPU 型号）
- 模型大小（1B vs 8B）
- `max_new_tokens` 设置
- 采样温度（高温度略慢）

---

## ❓ 常见问题

### Q1: 输出文件过大怎么办？

**A**: 使用压缩或减少采样次数

```bash
# 生成后压缩
gzip output.jsonl  # 压缩率通常 70-80%

# 或减少采样
--num-samples 5  # 而非 10
```

### Q2: 程序卡住不动？

**A**: 可能是某个样本的 prompt 过长，导致生成很慢

```bash
# 检查进度
# 查看 .checkpoint 文件最后一行
tail -1 output.jsonl.checkpoint

# 设置更小的 max_new_tokens
--max-new-tokens 64
```

### Q3: 采样结果质量差？

**A**: 调整温度和 top-p

```bash
# 提高质量（降低温度）
--temperature 0.7

# 提高多样性（升高温度）
--temperature 0.9

# 调整 top-p（默认 0.9 已经很好）
--top-p 0.95
```

### Q4: 如何验证采样是否随机？

**A**: 检查采样结果的差异性

```python
import json

with open("output.jsonl", 'r') as f:
    item = json.loads(f.readline())
    samples = item["stochastic_samples"]

    # 检查唯一性
    unique_ratio = len(set(samples)) / len(samples)
    print(f"唯一采样比例: {unique_ratio:.2%}")
    # 应该 > 80% 表示多样性足够

    # 打印前 3 个采样
    for i, s in enumerate(samples[:3], 1):
        print(f"Sample {i}: {s}")
```

---

## 📝 TODO

未来可能的改进：

- [ ] 支持批量生成（batch generation）提升速度
- [ ] 支持分布式采样（多 GPU）
- [ ] 添加采样质量评估（自动计算多样性指标）
- [ ] 支持从 HDF5 直接读取 prompt（避免重复读取）

---

## 🆘 技术支持

如遇问题，请检查：
1. 输入文件是否包含 `prompt` 字段
2. 模型是否正确加载（检查 `--trust-remote-code`）
3. 显存是否充足（使用 `nvidia-smi` 查看）
4. 检查点文件是否损坏（删除后重试）

---

**最后更新**: 2026-03-19
