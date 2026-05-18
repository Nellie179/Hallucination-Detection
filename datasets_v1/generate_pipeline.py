import os
import time

from prepare_datasets import process_dataset
from hidden_state import HiddenStateExtractor
from llm_judge import run_llm_judge

CONFIG = {
    "dataset_name": "ragtruth",
    "dataset_split": "train",
    "max_samples": 10000,

    "target_model": "meta-llama/Llama-3.2-3B-Instruct",
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 4,
    "max_new_tokens": 2048,

    "model_kwargs": {
        "trust_remote_code": True,
        "attn_implementation": "sdpa"
    },

    "generation_kwargs": {
        "do_sample": False,
    },

    "template_kwargs": {
        "enable_thinking": False
    },

    "layer_config": {"mode": "middle", "count": 5},
    "token_config": {"mode": "backward", "count": 5},
    "queue_size": 10,

    "judge_model": "gpt-4o-mini",
    "judge_concurrency": 2,

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


def run_pipeline():
    print("=" * 50)
    print("🚀 Initializing end-to-end hallucination detection pipeline (Universal Schema Edition)")
    print(f"📊 Dataset: {CONFIG['dataset_name']} | Max samples: {CONFIG['max_samples']}")
    print(f"🧠 Target model: {CONFIG['target_model']} (Few-shot: {CONFIG['num_shots']})")
    print(f"秤 Judge model: {CONFIG['judge_model']}")
    print("=" * 50 + "\n")

    paths = build_paths(CONFIG)

    print("\n>>> [Step 1/3] Executing structured dataset extraction step...")
    if os.path.exists(paths["step1_unified_jsonl"]):
        print(f"⏭️  Detected existing structured data file at target destination: {paths['step1_unified_jsonl']}. Skipping step.")
    else:
        out_path = process_dataset(
            adapter_name=CONFIG["dataset_name"],
            output_dir=os.path.dirname(paths["step1_unified_jsonl"]),
            split=CONFIG["dataset_split"],
            max_samples=CONFIG["max_samples"]
        )
        if os.path.exists(out_path):
            os.rename(out_path, paths["step1_unified_jsonl"])

    print("\n>>> [Step 2/3] Initiating contextual prompt generation and internal hidden state extraction...")
    if os.path.exists(paths["step2_tensor_h5"]) and os.path.exists(paths["step2_metadata_jsonl"]):
        print(f"⏭️  Detected existing HDF5 model activations and system metadata records. Skipping GPU extraction pass.")
    else:
        extractor = HiddenStateExtractor(
            model_name=CONFIG["target_model"],
            model_kwargs=CONFIG.get("model_kwargs", {})
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
            generation_kwargs=CONFIG.get("generation_kwargs", {}),
            template_kwargs=CONFIG.get("template_kwargs", {})
        )

        del extractor
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n>>> [Step 3/3] Deploying high-concurrency external LLM judge verification routine...")
    if os.path.exists(paths["step3_scored_jsonl"]):
        print(f"⏭️  Detected complete target evaluation file records: {paths['step3_scored_jsonl']}. Pipeline execution terminating early.")
    else:
        run_llm_judge(
            input_filepath=paths["step2_metadata_jsonl"],
            output_filepath=paths["step3_scored_jsonl"],
            failed_filepath=paths["step3_failed_jsonl"],
            model_name=CONFIG["judge_model"],
            concurrency_limit=CONFIG["judge_concurrency"]
        )

    print("\n" + "=" * 50)
    print("✅ Pipeline execution cycle successfully finished.")
    print(f"📁 Target core experiment assets stored in: {os.path.dirname(paths['step2_tensor_h5'])}")
    print("=" * 50)


if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    print(f"⏳ Total operational duration: {time.time() - start_time:.2f} seconds")