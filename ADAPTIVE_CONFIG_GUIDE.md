# 🧠 自适应配置系统使用指南

## 📖 概述

自适应配置系统能够自动检测您的硬件（GPU 架构、显存）和目标模型特性，智能推荐最优的运行参数，解决以下问题：

- ✅ **自动选择精度**: RTX 8000 使用 `float16`，A100 使用 `bfloat16`
- ✅ **自动选择注意力机制**: `sdpa` vs `eager` vs `flash_attention`
- ✅ **自动识别模型特性**: Qwen/DeepSeek 自动开启 `enable_thinking`
- ✅ **避免常见警告**: 自动清理无效参数，避免 bfloat16/flex_attention 警告

---

## 🚀 快速开始

### 方式 1: 完全自动（推荐）

在 `generate_pipeline.py` 中，将 kwargs 留空即可：

```python
CONFIG = {
    "target_model": "Qwen/Qwen3-8B",

    # 完全留空，系统会自动配置
    "model_kwargs": {},
    "generation_kwargs": {},
    "template_kwargs": {},
}
```

系统会自动生成：
```python
# 对于 Qwen3-8B + RTX 8000
model_kwargs = {
    'device_map': 'auto',
    'trust_remote_code': True,
    'torch_dtype': 'float16',  # 自动选择适合 RTX 8000 的精度
    'attn_implementation': 'sdpa'  # 自动选择高效的注意力实现
}
generation_kwargs = {'do_sample': False}
template_kwargs = {'enable_thinking': True}  # Qwen 自动开启
```

---

### 方式 2: 部分覆盖

你可以只覆盖你需要的参数，其他参数仍然自动配置：

```python
CONFIG = {
    "target_model": "Qwen/Qwen3-8B",

    "model_kwargs": {
        # 只覆盖这一个参数，其他仍自动配置
        "attn_implementation": "eager"
    },

    "generation_kwargs": {
        "do_sample": True,
        "temperature": 0.7
    },

    "template_kwargs": {},  # 仍然自动开启 enable_thinking
}
```

---

## 🎯 支持的模型家族

| 模型家族 | 自动配置 | 说明 |
|---------|---------|------|
| **Qwen** | `trust_remote_code=True`<br>`enable_thinking=True` | Qwen3+ 自动开启内部 CoT |
| **DeepSeek** | `trust_remote_code=True`<br>`enable_thinking=True` | 支持推理时思考 |
| **Llama** | `trust_remote_code=False` | 官方模型，无需信任代码 |
| **Mistral** | `trust_remote_code=False` | 官方模型 |
| **Gemma** | `trust_remote_code=False` | Google 官方模型 |
| **其他** | `trust_remote_code=True` | 保守策略：默认信任 |

---

## 🖥️ 硬件自适应

### GPU 架构检测

| GPU 型号 | 架构类别 | 推荐精度 | 注意力机制 |
|---------|---------|---------|-----------|
| A100, H100, RTX 30/40 | Ampere+ | `bfloat16` | `sdpa` |
| RTX 8000, RTX 20, V100 | Turing/Volta | `float16` | `sdpa` |
| GTX 10, Tesla K | Legacy | `float16` | `eager` |

### 自动避免的问题

1. **bfloat16 警告**: RTX 8000 自动使用 `float16`
2. **flex_attention 警告**: 自动改用 `sdpa`
3. **无效采样参数**: `do_sample=False` 时自动移除 `temperature`

---

## 💡 高级用法

### 在代码中直接使用

```python
from adaptive_config import get_adaptive_config

# 获取自适应配置
config = get_adaptive_config(
    model_name="Qwen/Qwen3-8B",
    user_model_kwargs={"load_in_8bit": True},  # 可选覆盖
    verbose=True  # 打印检测结果
)

# 使用配置
from hidden_state import HiddenStateExtractor
extractor = HiddenStateExtractor(
    model_name="Qwen/Qwen3-8B",
    model_kwargs=config["model_kwargs"]
)
```

### 添加新的模型家族

编辑 `adaptive_config.py` 中的 `MODEL_SPECIFIC_CONFIGS`:

```python
MODEL_SPECIFIC_CONFIGS = {
    "your_model": {
        "template_kwargs": {"enable_thinking": True},
        "trust_remote_code": True
    },
    ...
}
```

---

## 📊 运行示例

当你运行 `generate_pipeline.py` 时，会看到：

```
============================================================
🔍 GPU 硬件检测结果
============================================================
GPU 名称: Quadro RTX 8000
架构类别: turing_volta
计算能力: SM 7.5
推荐精度: float16
显存容量: 47.6 GB
Flash Attention 支持: ✓
============================================================

============================================================
⚙️  自适应配置生成结果
============================================================
目标模型: Qwen/Qwen3-8B
模型家族: Qwen

推荐配置:
  model_kwargs: {'device_map': 'auto', 'trust_remote_code': True,
                'torch_dtype': 'float16', 'attn_implementation': 'sdpa'}
  generation_kwargs: {'do_sample': False}
  template_kwargs: {'enable_thinking': True}
============================================================
```

---

## ❓ 常见问题

### Q: 我想强制使用 bfloat16 怎么办？

A: 在 `model_kwargs` 中手动指定即可（会收到警告但仍然执行）：

```python
"model_kwargs": {
    "torch_dtype": "bfloat16"
}
```

### Q: 不同模型需要不同配置怎么办？

A: 每次运行前修改 `CONFIG["target_model"]` 即可，系统会自动适配。

### Q: 如何禁用自适应配置？

A: 在 `model_kwargs` 中手动指定所有参数即可完全覆盖。

---

## 🔧 测试自适应配置

运行单元测试：

```bash
conda activate benchmark
cd /home/zfang1/Data/Lxy/Benchmark/datasets
python adaptive_config.py
```

---

## 📝 总结

**推荐做法**: 保持 `model_kwargs/generation_kwargs/template_kwargs` 为空字典，让系统自动配置。只在需要特殊配置时手动覆盖特定参数。

这样您的代码可以无缝适配：
- ✅ 不同的 GPU（RTX 8000 → A100 → H100）
- ✅ 不同的模型（Qwen → Llama → Mistral）
- ✅ 避免所有警告和性能陷阱
