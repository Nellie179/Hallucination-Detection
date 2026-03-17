import os
import time

from prepare_datasets import process_dataset
from hidden_state import HiddenStateExtractor
from llm_judge import run_llm_judge
from adaptive_config import get_adaptive_config

# ==========================================
# 💎 全局实验配置中枢 (唯一定义参数的地方)
# ==========================================
CONFIG = {
    # 1. 数据集配置
    "dataset_name": "coqa",
    "dataset_split": "train",
    "max_samples": 7000,

    # 2. 待测模型与 Prompt 渲染配置
    "target_model": "meta-llama/Meta-Llama-3.1-8B",  # 替换为你想测试的任意模型
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 4,
    "max_new_tokens": 2048,  # 增加到 256 以避免 thinking 模式下答案被截断

    # ------------------------------------------------------
    # 🌟 自由超参数透传空间 (kwargs)
    # ------------------------------------------------------
    # 【新特性】：现在支持智能自适应配置！
    # 留空则完全由系统自动检测 GPU 和模型特性，推荐最优配置
    # 如需手动覆盖某些参数，直接在下方字典中指定即可
    "model_kwargs": {
        # "trust_remote_code": True,  # 自动根据模型判断
        # "torch_dtype": "float16",   # 自动根据 GPU 架构选择
        # "attn_implementation": "sdpa",  # 自动选择最优注意力实现
    },

    "generation_kwargs": {
        # "do_sample": False,  # 默认贪婪解码
    },

    "template_kwargs": {
        "enable_thinking": False  # 自动根据模型类型判断（Qwen/DeepSeek 会自动开启）
    },
    # ------------------------------------------------------

    # 3. 张量提取配置 (GPU)
    # "layer_config": {"mode": "middle", "count": 5},
    "layer_config": [-5, -4, -3, -2, -1],
    "token_config": {"mode": "backward", "count": 5},
    "queue_size": 20,
    "batch_size": 16,  # 批处理大小：同时处理的样本数（建议 2-32，根据显存调整）
    "use_torch_compile": True,  # 启用 torch.compile() 加速（PyTorch 2.0+）

    # 4. 裁判模型配置 (API)
    "judge_model": "gpt-4o-mini",
    "judge_concurrency": 2,

    # 5. 存储路径规划
    "base_output_dir": "./experiments"
}


def build_paths(cfg):
    exp_dir = os.path.join(
        cfg["base_output_dir"],
        cfg["target_model"],
        f"{cfg['dataset_name']}_{cfg['max_samples']}samples"
    )
    os.makedirs(exp_dir, exist_ok=True)

    return {
        "step1_unified_jsonl": os.path.join(exp_dir, "01_structured_data.jsonl"),
        "step2_tensor_h5": os.path.join(exp_dir, "02_hidden_states.h5"),
        "step2_metadata_jsonl": os.path.join(exp_dir, "02_extracted_metadata.jsonl"),
        "step3_scored_jsonl": os.path.join(exp_dir, "03_final_scored_metadata.jsonl"),
        "step3_failed_jsonl": os.path.join(exp_dir, "03_judge_failed.jsonl"),
    }


# ==========================================
# 🚀 自动化流水线主引擎
# ==========================================
def run_pipeline():
    print("=" * 50)
    print("🚀 开始执行端到端幻觉检测流水线 (Universal Schema 版)")
    print(f"📊 数据集: {CONFIG['dataset_name']} | 样本数: {CONFIG['max_samples']}")
    print(f"🧠 待测模型: {CONFIG['target_model']} (Few-shot: {CONFIG['num_shots']})")
    print(f"⚖️ 裁判模型: {CONFIG['judge_model']}")
    print("=" * 50 + "\n")

    paths = build_paths(CONFIG)

    print("\n>>> [Step 1/3] 执行纯净结构化数据提取...")
    if os.path.exists(paths["step1_unified_jsonl"]):
        print(f"⏭️  检测到结构化数据已存在: {paths['step1_unified_jsonl']}，跳过生成。")
    else:
        out_path = process_dataset(
            adapter_name=CONFIG["dataset_name"],
            output_dir=os.path.dirname(paths["step1_unified_jsonl"]),
            split=CONFIG["dataset_split"],
            max_samples=CONFIG["max_samples"]
        )
        if os.path.exists(out_path):
            os.rename(out_path, paths["step1_unified_jsonl"])

    print("\n>>> [Step 2/3] 执行智能 Prompt 渲染与隐藏层提取 (GPU重负载)...")
    if os.path.exists(paths["step2_tensor_h5"]) and os.path.exists(paths["step2_metadata_jsonl"]):
        print(f"⏭️  检测到 HDF5 张量和 Metadata 已存在，跳过 GPU 提取，节省算力！")
    else:
        # 🚀 自适应配置生成：根据硬件和模型自动优化参数
        adaptive_config = get_adaptive_config(
            model_name=CONFIG["target_model"],
            user_model_kwargs=CONFIG.get("model_kwargs"),
            user_generation_kwargs=CONFIG.get("generation_kwargs"),
            user_template_kwargs=CONFIG.get("template_kwargs"),
            verbose=True
        )

        extractor = HiddenStateExtractor(
            model_name=CONFIG["target_model"],
            model_kwargs=adaptive_config["model_kwargs"],
            use_compile=CONFIG["use_torch_compile"]  # 传入 torch.compile 配置
        )
        extractor.process_from_file(
            input_jsonl_path=paths["step1_unified_jsonl"],
            output_h5_path=paths["step2_tensor_h5"],
            output_jsonl_path=paths["step2_metadata_jsonl"],
            layer_config=CONFIG["layer_config"],
            token_config=CONFIG["token_config"],
            max_new_tokens=CONFIG["max_new_tokens"],
            max_queue_size=CONFIG["queue_size"],
            system_prompt=CONFIG["system_prompt"],
            num_shots=CONFIG["num_shots"],
            generation_kwargs=adaptive_config["generation_kwargs"],
            template_kwargs=adaptive_config["template_kwargs"],
            batch_size=CONFIG["batch_size"]  # 启用批处理加速
        )

        del extractor
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n>>> [Step 3/3] 执行高并发 LLM 裁判打分...")
    if os.path.exists(paths["step3_scored_jsonl"]):
        print(f"⏭️  检测到最终打分文件已存在: {paths['step3_scored_jsonl']}，流水线提前结束！")
    else:
        run_llm_judge(
            input_filepath=paths["step2_metadata_jsonl"],
            output_filepath=paths["step3_scored_jsonl"],
            failed_filepath=paths["step3_failed_jsonl"],
            model_name=CONFIG["judge_model"],
            concurrency_limit=CONFIG["judge_concurrency"]
        )

    print("\n" + "=" * 50)
    print("✅ 流水线全部执行完毕！")
    print(f"📁 核心产物位于: {os.path.dirname(paths['step2_tensor_h5'])}")
    print("=" * 50)


if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    print(f"⏳ 总耗时: {time.time() - start_time:.2f} 秒")