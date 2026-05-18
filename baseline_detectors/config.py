import os

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "experiments"
)

DEFAULT_MODEL = os.getenv("BENCHMARK_MODEL", "Qwen/Qwen3-8B")
DEFAULT_DATASET = os.getenv("BENCHMARK_DATASET", "coqa_5000samples")  

EXPERIMENT_DIR = os.path.join(DATA_ROOT, DEFAULT_MODEL, DEFAULT_DATASET)

METADATA_JSONL = os.path.join(EXPERIMENT_DIR, "03_final_scored_metadata.jsonl")
HIDDEN_STATES_H5 = os.path.join(EXPERIMENT_DIR, "02_hidden_states.h5")

TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.7"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

SAMPLING_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "sampling_cache")
AUTO_GENERATE_SAMPLES = True

DEFAULT_SAMPLING_CONFIG = {
    "num_samples": 10,
    "temperature": 0.8,
    "top_p": 0.9
}

ACTIVE_DETECTORS = [
    {
        "registry_name": "saplma_linear_probe",
        "kwargs": {
            "target_layer_idx": -1,  
            "pooling": "mean"        
        }
    },
    {
        "registry_name": "sar",
        "kwargs": {
            "target_layer_idx": -1,       
            "normalize": True,            
            "use_effective_rank": True    
        }
    },
    {
        "registry_name": "ln_entropy",
        "kwargs": {
            "num_layers": 5,              
            "layer_selection": "last",    
            "entropy_type": "svd",        
            "normalize_per_layer": True
        }
    },
    {
        "registry_name": "perplexity",
        "kwargs": {
            "use_log_perplexity": False,  
            "clip_min_logprob": -100.0    
        }
    },
    {
        "registry_name": "lexical_similarity",
        "kwargs": {
            "metric": "jaccard",  
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
    {
        "registry_name": "selfcheck_bertscore",
        "kwargs": {
            "num_samples": 10,
            "temperature": 0.8,
            "bert_model": "microsoft/deberta-xlarge-mnli"
        }
    },
]

OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "benchmark_results")
SAVE_DETAILED_PREDICTIONS = True
VERBOSE = True


def validate_config():
    errors = []

    if not os.path.exists(METADATA_JSONL):
        errors.append(f"Metadata file not found: {METADATA_JSONL}")

    if not os.path.exists(HIDDEN_STATES_H5):
        errors.append(f"Hidden States file not found: {HIDDEN_STATES_H5}")

    if not (0.0 <= TRAIN_RATIO <= 1.0):
        errors.append(f"TRAIN_RATIO must be between 0.0 and 1.0, current value: {TRAIN_RATIO}")

    if errors:
        print("\n".join(errors))
        print(f"\nSuggestion: Please run datasets/generate_pipeline.py to generate target files.")
        print(f"Current search directory: {EXPERIMENT_DIR}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("📋 Benchmark Configuration Metadata")
    print("=" * 70)
    print(f"Data Root: {DATA_ROOT}")
    print(f"Experiment Directory: {EXPERIMENT_DIR}")
    print(f"Metadata: {METADATA_JSONL}")
    print(f"Hidden States: {HIDDEN_STATES_H5}")
    print(f"Train Ratio: {TRAIN_RATIO}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Active Detectors: {len(ACTIVE_DETECTORS)}")
    print("=" * 70)

    if validate_config():
        print("Config validation passed successfully.")
    else:
        print("Config validation failed.")