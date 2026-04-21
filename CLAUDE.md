# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the full benchmark (after data is prepared)
uv run python baseline_detectors/runner.py

# Run the dataset generation pipeline (first time / new model or dataset)
uv run python datasets_v1/generate_pipeline.py

# Run a specific data generation step
uv run python datasets_v1/generate_stochastic_samples.py
uv run python baseline_detectors/extract_full_hidden_states.py
```

## Architecture

The project is a hallucination detection benchmark. Data flows through three phases:

```
Raw JSONL metadata
  → [Phase 1] LLM inference + hidden state extraction → 02_hidden_states.h5
  → [Phase 2] Stochastic sample generation + auxiliary evals → 04_*.jsonl
  → [Phase 3] Detector scoring + evaluation → 06_evaluation_results.json
```

**`datasets_v1/`** owns Phases 1–2: downloading/formatting datasets, running the target LLM, extracting hidden states, generating multiple stochastic samples, and labeling outputs with a GPT-4 judge (`llm_judge.py`).

**`baseline_detectors/`** owns Phase 3: loading the cached data, fitting each detector on the train split, scoring the test split, and computing AUROC/AUPR/FPR@95.

Experiment outputs live under `experiments/{model}/{dataset}_Nsamples/` with numbered file prefixes (`01_`, `02_`, ...) that encode the pipeline stage.

## Detector System

### Registration

Detectors self-register via `@register_detector("name")` in `detectors/registry.py`. Every `.py` in `detectors/` is auto-imported by `detectors/__init__.py`, so a new detector file is picked up automatically — no manual registration needed.

### Base class & dependency flags

`BaseDetector` (detectors/base.py) exposes three boolean flags that the runner uses to decide what data to pre-generate:

| Flag | Meaning |
|------|---------|
| `requires_stochastic` | Needs multiple sampled outputs (texts + logprobs) |
| `requires_stochastic_hidden_states` | Also needs hidden states for those samples |
| `requires_qa_features` | Needs a second inference pass with "Prompt+Answer" concatenated |

Detectors with `requires_qa_features=True` get a dedicated H5 file: `05_qa_features_{detector_name}.h5`.

### Adding a new detector

1. Create `baseline_detectors/detectors/my_detector.py`
2. Inherit `BaseDetector`, set the dependency flags in `__init__`, implement `predict_score(self, accessor) -> float`
3. Add it to `ACTIVE_DETECTORS` in `baseline_detectors/config.py`

### Detector categories

- **Whitebox** (require hidden states): `saplma`, `sar`, `ccs`, `prism`, `eigenscore`, `icr_probe`, `sep`, `mind`, `haloscope`
- **Blackbox** (text/logprob only): `perplexity`, `lexical_similarity`, `verbalize`, `self_evaluator`
- **Sampling-based**: `selfcheck_bertscore`, `selfcheck_nli`, `semantic_entropy`

## Data Access

`SampleAccessor` (data_utils/accessor.py) is the single facade all detectors use. It wraps:
- `metadata` dict (prompt, output, `eval_category` ∈ `{"correct", "hallucination"}`)
- `h5_group` — original generation hidden states
- `stochastic_samples_dict` — multi-sample texts and sequence-level log-probs
- `stochastic_h5_group` — hidden states for stochastic samples
- `qa_h5_file` — Q+A paired hidden states (CCS, PRISM, ICR, SEP)

Key methods: `get_hidden_states(layer_idx)`, `get_qa_hidden_states(layer_idx)`, `get_stochastic_samples()`, `get_token_logprobs()`, `get_contrast_hidden_states()`.

## Configuration

`baseline_detectors/config.py` sets `DATA_ROOT`, `DEFAULT_MODEL`, `DEFAULT_DATASET`, `TRAIN_RATIO`, `OUTPUT_DIR`, and `ACTIVE_DETECTORS`. `runner.py` contains `EVAL_CONFIG` for generation hyperparameters (temperature, num_samples, layer/token selection).

The runner auto-validates caches at startup and only re-generates missing data.

## Environment

Requires Python 3.12. GPU inference runs on Linux with CUDA 11.8 (PyTorch `cu118` index configured in `pyproject.toml`). Set `OPENAI_API_KEY` for the GPT-4 judge in `datasets_v1/llm_judge.py`.
