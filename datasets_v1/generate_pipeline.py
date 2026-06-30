import os
import time
import argparse

from prepare_datasets import process_dataset
from hidden_state import HiddenStateExtractor
from llm_judge import run_llm_judge

# Default configuration. Any field here can be overridden from the command line
# (see parse_args below). Running `python generate_pipeline.py` with no arguments
# reproduces the original behavior exactly.
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


def run_pipeline(cfg=CONFIG):
    print("=" * 50)
    print("🚀 Initializing end-to-end hallucination detection pipeline (Universal Schema Edition)")
    print(f"📊 Dataset: {cfg['dataset_name']} | Max samples: {cfg['max_samples']}")
    print(f"🧠 Target model: {cfg['target_model']} (Few-shot: {cfg['num_shots']})")
    print(f"秤 Judge model: {cfg['judge_model']}")
    print("=" * 50 + "\n")

    paths = build_paths(cfg)

    print("\n>>> [Step 1/3] Executing structured dataset extraction step...")
    if os.path.exists(paths["step1_unified_jsonl"]):
        print(f"⏭️  Detected existing structured data file at target destination: {paths['step1_unified_jsonl']}. Skipping step.")
    else:
        out_path = process_dataset(
            adapter_name=cfg["dataset_name"],
            output_dir=os.path.dirname(paths["step1_unified_jsonl"]),
            split=cfg["dataset_split"],
            max_samples=cfg["max_samples"]
        )
        if os.path.exists(out_path):
            os.rename(out_path, paths["step1_unified_jsonl"])

    print("\n>>> [Step 2/3] Initiating contextual prompt generation and internal hidden state extraction...")
    if os.path.exists(paths["step2_tensor_h5"]) and os.path.exists(paths["step2_metadata_jsonl"]):
        print(f"⏭️  Detected existing HDF5 model activations and system metadata records. Skipping GPU extraction pass.")
    else:
        extractor = HiddenStateExtractor(
            model_name=cfg["target_model"],
            model_kwargs=cfg.get("model_kwargs", {})
        )
        extractor.process_from_file(
            input_jsonl_path=paths["step1_unified_jsonl"],
            output_h5_path=paths["step2_tensor_h5"],
            output_jsonl_path=paths["step2_metadata_jsonl"],
            layer_config=cfg["layer_config"],
            token_config=cfg["token_config"],
            max_new_tokens=cfg["max_new_tokens"],
            max_queue_size=cfg["queue_size"],
            system_prompt=cfg["system_prompt"],
            num_shots=cfg["num_shots"],
            generation_kwargs=cfg.get("generation_kwargs", {}),
            template_kwargs=cfg.get("template_kwargs", {})
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
            model_name=cfg["judge_model"],
            concurrency_limit=cfg["judge_concurrency"]
        )

    print("\n" + "=" * 50)
    print("✅ Pipeline execution cycle successfully finished.")
    print(f"📁 Target core experiment assets stored in: {os.path.dirname(paths['step2_tensor_h5'])}")
    print("=" * 50)


def parse_args():
    """Parse command-line overrides. Any argument left unset falls back to the
    corresponding default in CONFIG, so `python generate_pipeline.py` with no
    arguments behaves exactly as before."""
    p = argparse.ArgumentParser(
        description="OpenHalDet — Phase 1: data preparation pipeline "
                    "(structuring → generation + hidden-state extraction → LLM-judge annotation)."
    )
    p.add_argument("--dataset", default=CONFIG["dataset_name"],
                   help="Dataset name / adapter key (e.g. ragtruth, triviaqa, gsm8k).")
    p.add_argument("--split", default=CONFIG["dataset_split"],
                   help="Dataset split to load (e.g. train, validation, test).")
    p.add_argument("--max_samples", type=int, default=CONFIG["max_samples"],
                   help="Maximum number of samples to process.")
    p.add_argument("--model", default=CONFIG["target_model"],
                   help="Backbone LLM (any HuggingFace AutoModelForCausalLM-compatible id).")
    p.add_argument("--num_shots", type=int, default=CONFIG["num_shots"],
                   help="Number of few-shot demonstrations.")
    p.add_argument("--max_new_tokens", type=int, default=CONFIG["max_new_tokens"],
                   help="Maximum number of newly generated tokens per response.")
    p.add_argument("--judge_model", default=CONFIG["judge_model"],
                   help="LLM-judge model used for truthfulness annotation.")
    p.add_argument("--judge_concurrency", type=int, default=CONFIG["judge_concurrency"],
                   help="Concurrency limit for the LLM-judge annotation stage.")
    p.add_argument("--output_dir", default=CONFIG["base_output_dir"],
                   help="Base directory for all generated artifacts.")
    return p.parse_args()


def build_config_from_args(args):
    """Merge CLI overrides onto the default CONFIG without mutating the original."""
    cfg = dict(CONFIG)
    cfg["dataset_name"] = args.dataset
    cfg["dataset_split"] = args.split
    cfg["max_samples"] = args.max_samples
    cfg["target_model"] = args.model
    cfg["num_shots"] = args.num_shots
    cfg["max_new_tokens"] = args.max_new_tokens
    cfg["judge_model"] = args.judge_model
    cfg["judge_concurrency"] = args.judge_concurrency
    cfg["base_output_dir"] = args.output_dir
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config_from_args(args)
    start_time = time.time()
    run_pipeline(cfg)
    print(f"⏳ Total operational duration: {time.time() - start_time:.2f} seconds")