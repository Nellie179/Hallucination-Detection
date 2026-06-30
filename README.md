# OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios

<p align="center">
  <a href="https://arxiv.org/abs/2606.06959"><img src="https://img.shields.io/badge/arXiv-2606.06959-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Nellie179/Hallucination-Detection"><img src="https://img.shields.io/badge/GitHub-Hallucination--Detection-blue?logo=github" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **OpenHalDet** is a unified benchmark for evaluating hallucination detection methods across diverse LLM generation scenarios. It standardizes the full evaluation pipeline — from prompt construction and response generation to truthfulness annotation, detector scoring, and metric computation — enabling fair, reproducible comparison across black-box, gray-box, and white-box detector families.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Pipeline Overview](#-pipeline-overview)
- [Step 1: Data Preparation](#-step-1-data-preparation)
- [Step 2: Running the Benchmark](#-step-2-running-the-benchmark)
- [Supported Datasets](#-supported-datasets)
- [Supported Detectors](#-supported-detectors)
- [Supported Backbone LLMs](#-supported-backbone-llms)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [Citation](#-citation)

---

## 🔍 Overview

Hallucination detection is critical for the reliable deployment of large language models (LLMs). Existing evaluations suffer from two core problems: **inconsistent inference and evaluation configurations**, and **limited coverage of downstream domains and tasks**. This makes reported detector performance difficult to compare, reproduce, or generalize.

**OpenHalDet** addresses these challenges by providing:

- A **standardized evaluation pipeline** covering 17 datasets across diverse generation scenarios
- A **unified detector interface** supporting 16 representative black-box, gray-box, and white-box methods
- A **decoupled architecture** that separates response generation, signal extraction, and detector scoring, enabling artifact reuse and controlled comparison
- An **extensible open-source codebase** that allows new datasets and detectors to be integrated without rebuilding the pipeline

---

## ✨ Key Features

- **17 datasets** spanning QA (multiple-choice, open-ended, reading comprehension, multi-hop, conversational, grounded), RAG, summarization, mathematical reasoning, scientific reasoning, code generation, agentic tool use, and multilingual evaluation
- **16 detectors** across three model-access regimes (black-box / gray-box / white-box)
- **5 backbone LLMs** from the Llama and Qwen families (3B to 70B)
- **Run from the command line** — pick datasets, backbones, and detectors with simple flags; no need to edit source files
- **Unified instance schema** that normalizes heterogeneous task formats into a common structured representation
- **GPT-4o-mini annotation** for scalable, reference-grounded truthfulness labeling
- **AUROC** as the primary metric; **Cost@N** for efficiency analysis
- **Resumable pipeline** with stage-level caching. Expensive GPU steps are skipped automatically when artifacts already exist

---

## 🛠 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: NVIDIA H100 / H800 / A100)
- A Hugging Face account (to download backbone models)
- An OpenAI API key (for GPT-4o-mini annotation in Step 1)

### 1. Clone and install dependencies

```bash
git clone https://github.com/Nellie179/Hallucination-Detection.git
cd Hallucination-Detection
pip install -r requirements.txt
```

### 2. Log in to Hugging Face

Backbone models are downloaded from the Hugging Face Hub. Log in once from your terminal so the download has access to your account:

```bash
huggingface-cli login
```

Paste an access token when prompted (create one at <https://huggingface.co/settings/tokens>).

> **Gated models (Llama family).** Llama-3.x checkpoints require one-time access approval. Open the model page (e.g. <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>) and click **Request access** before downloading; approval is usually quick. **Qwen3 models (Apache 2.0) are not gated** and work immediately after login, so they are the easiest choice to get started.

### 3. Set your OpenAI API key

Step 1 uses `gpt-4o-mini` to annotate responses, so an OpenAI API key is required (create one at <https://platform.openai.com/api-keys>; this stage makes paid API calls).

Create a file at `datasets_v1/.env` and add:

```
OPENAI_API_KEY=your_openai_key_here
OPENAI_BASE_URL="https://api.openai.com/v1"
```

> Keep this file private. Add `.env` to your `.gitignore` so your key is never committed.

---

## 🔄 Pipeline Overview

The benchmark runs in two independent phases — two commands end to end:

```bash
# Phase 1 — prepare data (structuring + generation + annotation)
python datasets_v1/generate_pipeline.py --dataset triviaqa --model meta-llama/Llama-3.2-3B-Instruct --max_samples 100

# Phase 2 — run detectors and report AUROC
python baseline_detectors/runner.py --models meta-llama/Llama-3.2-3B-Instruct --datasets triviaqa --baselines mind prism perplexity
```

What each phase does:

```
Phase 1 — Data Preparation (datasets_v1/)
┌───────────────────────────────────────────────────────────┐
│  Step 1: Dataset structuring   →  01_structured_data.jsonl │
│  Step 2: LLM generation +      →  02_hidden_states.h5      │
│          hidden-state extract      02_extracted_metadata.jsonl│
│  Step 3: LLM-judge annotation  →  03_final_scored_metadata.jsonl│
└───────────────────────────────────────────────────────────┘

Phase 2 — Detector Evaluation (baseline_detectors/)
┌───────────────────────────────────────────────────────────┐
│  Train/val/test split                                      │
│  Stochastic sampling (for sample-based detectors)         │
│  Auxiliary evaluations (for verbalize / self-evaluator)   │
│  QA feature extraction (for white-box detectors)          │
│  Detector fitting + scoring  →  benchmark_results/        │
└───────────────────────────────────────────────────────────┘
```

---

## 📦 Step 1: Data Preparation

Run the three-stage data pipeline (dataset structuring → LLM generation + hidden-state extraction → LLM-judge annotation) with a single command:

```bash
cd datasets_v1
python generate_pipeline.py --dataset triviaqa --model meta-llama/Llama-3.2-3B-Instruct --max_samples 100
```

Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `ragtruth` | Dataset name / adapter key (see supported list below) |
| `--model` | `meta-llama/Llama-3.2-3B-Instruct` | Backbone LLM (any HF `AutoModelForCausalLM` id) |
| `--max_samples` | `10000` | Maximum number of samples to process |
| `--split` | `train` | Dataset split to load |
| `--num_shots` | `4` | Number of few-shot demonstrations |
| `--judge_model` | `gpt-4o-mini` | LLM-judge model for annotation |
| `--output_dir` | `./experiments` | Base directory for generated artifacts |

> Running `python generate_pipeline.py` with no flags reproduces the default configuration. For advanced options (hidden-state layer/token selection, decoding kwargs, etc.), edit the `CONFIG` block at the top of `generate_pipeline.py`.

This produces the following artifacts under `experiments/<model>/<dataset>_<N>samples/`:

| File | Description |
|------|-------------|
| `01_structured_data.jsonl` | Dataset instances in the unified schema |
| `02_hidden_states.h5` | Hidden states (HDF5) for selected layers/tokens |
| `02_extracted_metadata.jsonl` | Per-sample metadata with generated responses |
| `03_final_scored_metadata.jsonl` | Annotated metadata with truthfulness labels |
| `03_judge_failed.jsonl` | Annotation failures for inspection or retry |

The pipeline is **resumable** — each stage checks for existing output files and skips completed work automatically.

---

## 🚀 Step 2: Running the Benchmark

Select your backbones, datasets, and detectors with command-line flags:

```bash
cd baseline_detectors
python runner.py \
  --models meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen3-8B \
  --datasets triviaqa gsm8k humaneval \
  --baselines verbalize perplexity mind prism
```

Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `Qwen/Qwen3-14B` | One or more backbone LLM ids (space-separated) |
| `--datasets` | `belebele` | One or more dataset keys (space-separated) |
| `--baselines` | *(all detectors)* | One or more detector registry keys; omit to run all 16+ detectors |
| `--seed` | `42` | Global random seed |

> Running `python runner.py` with no flags reproduces the default run configuration. Detector registry keys are listed in the [Supported Detectors](#-supported-detectors) section below.

Results (AUROC per detector per dataset) are saved to `experiments/<model>/<dataset>/benchmark_results/`.

---

## 📊 Supported Datasets

| Scenario | Dataset | Task Format |
|----------|---------|-------------|
| QA: Multiple-choice | ARC-Challenge, CommonsenseQA | MCQ |
| QA: Open-ended | TriviaQA, TruthfulQA | Short answer |
| QA: Reading comprehension | SQuAD v2 | Span extraction |
| QA: Multi-hop | HotpotQA | Cross-document reasoning |
| QA: Conversational | CoQA | Dialogue |
| QA: Grounded | HaluEval-QA | Context-based QA |
| Retrieval-augmented generation | RAGTruth | RAG generation |
| Summarization | XSum | Abstractive summarization |
| Mathematical reasoning | GSM8K, SVAMP | Chain-of-thought |
| Scientific reasoning | TheoremQA | Chain-of-thought |
| Code generation | HumanEval, MBPP | Code synthesis |
| Agentic tasks | xLAM-Agent | Tool invocation |
| Multilingual evaluation | Belebele | Multilingual MCQ |

To add a new dataset, create an adapter class in `datasets_v1/prepare_datasets.py` that inherits from `BaseDatasetAdapter` and implement `extract_structured_data()`, mapping raw fields to the unified schema:

```python
class MyDatasetAdapter(BaseDatasetAdapter):
    dataset_path = "hf-org/my-dataset"

    def extract_structured_data(self, row):
        return {
            "task_type":          "qa",
            "system_instruction": "",
            "context":            row.get("context", ""),
            "question":           row["question"],
            "choices":            {},
            "ground_truths":      [row["answer"]],
            "incorrect_answers":  []
        }
```

---

## 🤖 Supported Detectors

### Black-box (text output only)

| Name | Registry Key | Reference |
|------|-------------|-----------|
| Verbalized Confidence | `verbalize` | Lin et al., TMLR'22 |
| SelfCheckGPT-BERTScore | `selfcheck_bertscore` | Manakul et al., EMNLP'23 |
| SelfCheckGPT-NLI | `selfcheck_nli` | Manakul et al., EMNLP'23 |
| Lexical Similarity | `lexical_similarity` | Lin et al., TMLR'24 |

### Gray-box (token probabilities / likelihoods)

| Name | Registry Key | Reference |
|------|-------------|-----------|
| Perplexity | `perplexity` | Ren et al., ICLR'23 |
| Self-Evaluation | `self_evaluator` | Kadavath et al., ArXiv'22 |
| LN-Entropy | `ln_entropy` | Malinin & Gales, ICLR'21 |
| SAR | `sar` | Duan et al., ACL'24 |
| Semantic Entropy | `semantic_entropy` | Farquhar et al., Nature'24 |

### White-box (hidden states / internal representations)

| Name | Registry Key | Reference |
|------|-------------|-----------|
| EigenScore | `eigenscore_internal` | Chen et al., ICLR'24 |
| CCS | `ccs` | Burns et al., ICLR'23 |
| HaloScope | `haloscope` | Du et al., NeurIPS'24 |
| SAPLMA | `saplma` | Azaria & Mitchell, EMNLP'23 |
| MIND | `mind` | Su et al., ACL'24 |
| SEP | `sep` | Kossen et al., ArXiv'24 |
| ICR Probe | `icr_probe` | Zhang et al., ACL'25 |
| PRISM | `prism` | Zhang et al., ACL'25 |

To add a new detector, implement `BaseDetector` and register it:

```python
from detectors.registry import register_detector
from detectors.base import BaseDetector

@register_detector("my_detector")
class MyDetector(BaseDetector):
    def fit(self, train_accessors):
        pass  # optional training step

    def predict_score(self, accessor) -> float:
        # return higher scores for higher hallucination risk
        ...
```

---

## 🧠 Supported Backbone LLMs

| Model | Family | Parameters |
|-------|--------|-----------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama | 8B |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama | 3B |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama | 70B |
| `Qwen/Qwen3-8B` | Qwen | 8B |
| `Qwen/Qwen3-14B` | Qwen | 14B |

Any Hugging Face `AutoModelForCausalLM`-compatible model can be used by updating `target_model` in the pipeline config.

---

## 🗂 Repository Structure

```
Hallucination-Detection/
│
├── datasets_v1/                        # Phase 1 — data preparation pipeline
│   ├── .env                            # OpenAI API credentials
│   ├── generate_pipeline.py            # ★ Main entry point for data prep
│   ├── prepare_datasets.py             # Dataset adapters (17+ datasets)
│   ├── hidden_state.py                 # LLM response generation + hidden-state extraction
│   ├── llm_judge.py                    # Async GPT-4o-mini annotation
│   ├── generate_stochastic_samples.py  # Stochastic response sampling
│   ├── generate_auxiliary_evals.py     # Verbalize / self-eval auxiliary outputs
│   └── prompt_builder.py              # Model-aware prompt construction
│
├── baseline_detectors/                 # Phase 2 — detector evaluation
│   ├── runner.py                       # ★ Main entry point for benchmarking
│   ├── config.py                       # Global configuration & validation
│   │
│   ├── detectors/                      # Detector implementations
│   │   ├── base.py                     # BaseDetector interface
│   │   ├── registry.py                 # Detector registry (register / build)
│   │   ├── verbalize.py
│   │   ├── selfcheck_bertscore.py
│   │   ├── selfcheck_nli.py
│   │   ├── lexical_similarity.py
│   │   ├── uncertainty_metrics.py      # Perplexity / LN-Entropy shared utilities
│   │   ├── sar.py
│   │   ├── semantic_entropy.py
│   │   ├── self_evaluator.py
│   │   ├── eigenscore.py
│   │   ├── ccs.py
│   │   ├── haloscope.py
│   │   ├── saplma.py
│   │   ├── mind.py
│   │   ├── sep.py
│   │   ├── icr_probe.py
│   │   └── prism.py
│   │
│   ├── data_utils/                     # Shared utilities for detectors
│   │   ├── accessor.py                 # SampleAccessor (unified data loader)
│   │   ├── data_split.py               # Stratified 60/20/20 train/val/test split
│   │   ├── extract_qa_hidden_states.py # White-box feature extraction
│   │   ├── icr_score.py                # ICR attention/residual scoring
│   │   ├── labels.py                   # Label parsing utilities
│   │   ├── llm_layers.py               # Layer selection helpers
│   │   ├── sampling_manager.py         # Stochastic sample management
│   │   ├── train_utils.py              # MLP / probe training helpers
│   │   └── cache_utils.py              # Artifact caching utilities
│   │
│   └── evaluators/
│       └── classification.py           # AUROC / AUPR / FPR@95TPR computation
│
├── experiments/                        # Auto-created; stores all artifacts
│   └── <model>/<dataset>_<N>samples/
│       ├── 01_structured_data.jsonl
│       ├── 02_hidden_states.h5
│       ├── 02_extracted_metadata.jsonl
│       ├── 03_final_scored_metadata.jsonl
│       ├── 03_train.jsonl / 03_val.jsonl / 03_test.jsonl
│       ├── 04_stochastic_samples.jsonl
│       ├── 04_auxiliary_evals.jsonl
│       └── benchmark_results/
│
└── requirements.txt
```

---

## 📈 Results

AUROC (%) aggregated by scenario on Llama and Qwen backbones. See the paper for full per-dataset results.

| Method | QA | RAG | Sum. | Math | Code | Agent | Multi. |
|--------|----|-----|------|------|------|-------|--------|
| **Black-box avg.** | 68.76 | 58.08 | 60.76 | 67.00 | 62.60 | 63.73 | 69.48 |
| **Gray-box avg.** | 69.07 | 57.66 | 56.48 | 67.82 | 65.33 | 65.68 | 73.36 |
| **White-box avg.** | 66.43 | 58.71 | 57.25 | 67.80 | 66.26 | 70.49 | 67.85 |

*Results shown for Llama-3.2-3B-Instruct. Higher is better.*

**Key findings:**

1. **Detector effectiveness is scenario- and backbone-dependent.** No single detector family uniformly dominates across all tasks and models.
2. **More model access does not guarantee better detection.** Gray-box methods are competitive with white-box methods despite requiring only token probabilities.
3. **Evidence acquisition dominates practical cost.** Sampling-based detectors are significantly more expensive; accuracy-only comparisons are incomplete.

---

## 📖 Citation

If you use OpenHalDet in your research, please cite:

```bibtex
@article{li2026openhaldet,
  title     = {OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios},
  author    = {Li, Xinyi and Fang, Zhen and Deng, Yongxin and Luo, Jinyuan and Ma, Hongnan and
               Oh, Changdae and Shi, Zijing and Ye, Shanshan and Wang, Hanchen and Chen, Shu-Lin and
               Luo, Yadan and Yang, Mengyue and Du, Sean and Li, Sharon and Chen, Ling},
  journal   = {arXiv preprint arXiv:2606.06959},
  year      = {2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE). Note that individual datasets and backbone models are subject to their own licenses — please refer to the original sources before use. Key third-party licenses include the Llama 3 Community License (Meta), Apache 2.0 (Qwen3), and various dataset-specific terms summarized in the paper's Appendix L.

---

## 🙏 Acknowledgements

We thank the authors of all baseline detectors and benchmark datasets included in OpenHalDet. This work was supported by researchers at the University of Technology Sydney, University of Wisconsin–Madison, University of Bristol, The University of Queensland, and Nanyang Technological University.
