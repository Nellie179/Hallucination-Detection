# baseline_detectors/config.py
"""
Benchmark 运行配置中心
"""

import os

# ==========================================
# 数据文件路径配置
# ==========================================

# 数据根目录（自动检测）
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "experiments"
)

# 默认数据集配置（可以通过环境变量覆盖）
DEFAULT_MODEL = os.getenv("BENCHMARK_MODEL", "Qwen/Qwen3-8B")
DEFAULT_DATASET = os.getenv("BENCHMARK_DATASET", "coqa_5000samples")  # 修改为实际存在的数据集

# 自动构建数据路径
EXPERIMENT_DIR = os.path.join(DATA_ROOT, DEFAULT_MODEL, DEFAULT_DATASET)

# 核心数据文件
METADATA_JSONL = os.path.join(EXPERIMENT_DIR, "03_final_scored_metadata.jsonl")
HIDDEN_STATES_H5 = os.path.join(EXPERIMENT_DIR, "02_hidden_states.h5")

# ==========================================
# 数据集划分配置
# ==========================================

# 训练集比例（0.0 - 1.0）
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.7"))

# 随机种子（保证可复现）
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ==========================================
# 采样数据配置（新增）
# ==========================================

# 采样数据缓存目录
SAMPLING_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "sampling_cache")

# 是否自动生成缺失的采样数据
AUTO_GENERATE_SAMPLES = True

# 默认采样配置
DEFAULT_SAMPLING_CONFIG = {
    "num_samples": 10,
    "temperature": 0.8,
    "top_p": 0.9
}

# ==========================================
# Detector 配置
# ==========================================

# 激活的 Detectors 列表
ACTIVE_DETECTORS = [
    # === Whitebox Detectors（需要隐藏状态） ===
    {
        "registry_name": "saplma_linear_probe",
        "kwargs": {
            "target_layer_idx": -1,  # 使用最后一层
            "pooling": "mean"        # 平均池化
        }
    },

    # === Whitebox Detectors（需要隐藏状态） - 新增 ===
    {
        "registry_name": "sar",
        "kwargs": {
            "target_layer_idx": -1,       # 使用最后一层
            "normalize": True,            # 归一化隐藏状态
            "use_effective_rank": True    # 使用有效秩
        }
    },
    {
        "registry_name": "ln_entropy",
        "kwargs": {
            "num_layers": 5,              # 分析最后5层
            "layer_selection": "last",    # 可选: last, uniform, all
            "entropy_type": "svd",        # 可选: svd, variance, token_std
            "normalize_per_layer": True
        }
    },

    # === Blackbox Detectors（免采样，立即可用） ===
    {
        "registry_name": "perplexity",
        "kwargs": {
            "use_log_perplexity": False,  # 是否使用 log(PPL)
            "clip_min_logprob": -100.0    # 最小对数概率
        }
    },
    {
        "registry_name": "lexical_similarity",
        "kwargs": {
            "metric": "jaccard",  # 可选: jaccard, dice, overlap, cosine
            "use_stemming": False
        }
    },
    {
        "registry_name": "verbalize",
        "kwargs": {
            "language": "english",
            "normalize": True,
            "case_sensitive": False
        }
    },

    # === Blackbox Detectors（需要采样数据） ===
    # 注意：首次运行会自动生成采样数据，需要较长时间
    {
        "registry_name": "selfcheck_bertscore",
        "kwargs": {
            "num_samples": 10,
            "temperature": 0.8,
            "bert_model": "microsoft/deberta-xlarge-mnli"
        }
    },
    # {
    #     "registry_name": "selfcheck_nli",
    #     "kwargs": {
    #         "num_samples": 10,
    #         "temperature": 0.8,
    #         "nli_model": "microsoft/deberta-v2-xlarge-mnli"
    #     }
    # },
    # {
    #     "registry_name": "semantic_entropy",
    #     "kwargs": {
    #         "num_samples": 20,       # 语义熵需要更多采样
    #         "temperature": 0.9,      # 更高温度增加多样性
    #         "embedding_model": "all-MiniLM-L6-v2",
    #         "clustering_eps": 0.3
    #     }
    # },
]

# ==========================================
# 输出配置
# ==========================================

# 结果保存目录
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "benchmark_results")

# 是否保存详细的预测结果
SAVE_DETAILED_PREDICTIONS = True

# 是否打印详细日志
VERBOSE = True


# ==========================================
# 运行时验证
# ==========================================

def validate_config():
    """
    验证配置的有效性，在运行前检查必要的文件是否存在
    """
    errors = []

    if not os.path.exists(METADATA_JSONL):
        errors.append(f"❌ Metadata 文件不存在: {METADATA_JSONL}")

    if not os.path.exists(HIDDEN_STATES_H5):
        errors.append(f"❌ Hidden States 文件不存在: {HIDDEN_STATES_H5}")

    if not (0.0 <= TRAIN_RATIO <= 1.0):
        errors.append(f"❌ TRAIN_RATIO 必须在 0.0-1.0 之间，当前值: {TRAIN_RATIO}")

    if errors:
        print("\n".join(errors))
        print(f"\n💡 提示: 请先运行 datasets/generate_pipeline.py 生成数据")
        print(f"   当前查找路径: {EXPERIMENT_DIR}")
        return False

    return True


if __name__ == "__main__":
    # 测试配置
    print("=" * 70)
    print("📋 Benchmark 配置信息")
    print("=" * 70)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"实验目录: {EXPERIMENT_DIR}")
    print(f"Metadata: {METADATA_JSONL}")
    print(f"Hidden States: {HIDDEN_STATES_H5}")
    print(f"训练集比例: {TRAIN_RATIO}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"激活的 Detectors: {len(ACTIVE_DETECTORS)} 个")
    print("=" * 70)

    if validate_config():
        print("✅ 配置验证通过！")
    else:
        print("❌ 配置验证失败！")
