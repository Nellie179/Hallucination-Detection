import os
import sys
import json
import h5py
import torch
import math
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import numpy as np


def set_global_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Global random seed locked to: {seed}")


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "datasets_v1")
utils_dir = os.path.join(project_root, "data_utils")

if project_root not in sys.path: sys.path.insert(0, project_root)
if data_dir not in sys.path: sys.path.insert(0, data_dir)
if utils_dir not in sys.path: sys.path.insert(0, utils_dir)

from datasets_v1.generate_stochastic_samples import StochasticExtractor
from datasets_v1.generate_auxiliary_evals import AuxiliaryEvaluator
from data_utils.accessor import SampleAccessor
from baseline_detectors.evaluators.classification import ClassificationEvaluator
from data_utils.extract_qa_hidden_states import process_dataset
from data_utils.data_split import split_dataset

ACADEMIC_KWARGS = {"num_beams": 5, "do_sample": True, "temperature": 1.0}
FAST_KWARGS = {"num_beams": 1, "do_sample": True, "temperature": 1.0, "top_p": 0.9}

EVAL_CONFIG = {
    "num_samples": 5,
    "max_new_tokens": 2048,
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 4,
    "layer_config": {"mode": "middle", "count": 5},
    "token_config": {"mode": "backward", "count": 5},

    "stochastic_gen_kwargs": FAST_KWARGS,
    "pooling_method": "mean",
    "template_kwargs": {"enable_thinking": False},
    "model_kwargs": {"trust_remote_code": True}
}


def enforce_safe_resume(base_meta_path, target_jsonl, target_h5, expected_total, rollback_count=2):
    ordered_items = []
    with open(base_meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): ordered_items.append(json.loads(line))

    valid_json_lines = []
    json_ids = set()
    if target_jsonl and os.path.exists(target_jsonl):
        with open(target_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    json_ids.add(str(data["sample_id"]))
                    valid_json_lines.append(line)
                except:
                    break

    h5_ids = set()
    if target_h5 and os.path.exists(target_h5):
        try:
            with h5py.File(target_h5, 'r') as f_h5:
                h5_keys = set(f_h5.keys())
                for item in ordered_items:
                    sid = str(item["sample_id"])
                    if sid in h5_keys or any(k.startswith(sid + "_") for k in h5_keys):
                        h5_ids.add(sid)
        except Exception:
            pass

    dlq_file = target_h5 + ".failed_ids.txt" if target_h5 else (
        target_jsonl + ".failed_ids.txt" if target_jsonl else None)
    failed_ids = set()
    if dlq_file and os.path.exists(dlq_file):
        with open(dlq_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    failed_ids.add(line.split('\t')[0].strip())

    if target_jsonl and target_h5:
        processed_ids = json_ids.intersection(h5_ids)
    elif target_jsonl:
        processed_ids = json_ids
    elif target_h5:
        processed_ids = h5_ids
    else:
        processed_ids = set()

    effective_ids = processed_ids.union(failed_ids)

    if len(effective_ids) == 0: return True

    highest_idx = -1
    for i, item in enumerate(ordered_items):
        if str(item["sample_id"]) in effective_ids:
            highest_idx = i

    if len(effective_ids) >= expected_total and highest_idx == expected_total - 1:
        if failed_ids:
            print(
                f"    [+] Benchmark progress 100% completed ({len(failed_ids)} items resolved via DLQ queue). Skipping routine execution.")
        return False

    if highest_idx == expected_total - 1:
        missing = expected_total - len(effective_ids)
        print(
            f"    🛠️ Progress reached terminal limit but discovered {missing} unaccounted entities. Engaging non-destructive target patch mode...")
        return True

    safe_limit_idx = max(-1, highest_idx - rollback_count)
    safe_ids = set([str(item["sample_id"]) for item in ordered_items[:safe_limit_idx + 1]])

    print(
        f"    🛠️ Processing interrupted mid-stream (Breakpoint detected at: {highest_idx + 1}/{expected_total}). Engaging step-back clearance (-{rollback_count})...")

    if target_jsonl and os.path.exists(target_jsonl):
        with open(target_jsonl, 'w', encoding='utf-8') as f:
            for line in valid_json_lines:
                try:
                    if str(json.loads(line)["sample_id"]) in safe_ids: f.write(line)
                except:
                    pass

    if target_h5 and os.path.exists(target_h5):
        try:
            with h5py.File(target_h5, 'a') as f_h5:
                for key in list(f_h5.keys()):
                    if not any(key == sid or key.startswith(sid + "_") for sid in safe_ids):
                        del f_h5[key]
        except Exception:
            pass

    print(
        f"    [+] Corruption cleanup verified. Engine automatically stepping past first {len(safe_ids)} files to seamlessly resume writing.")
    return True


def get_detector(name: str):
    detectors_map = {
        "selfcheck_nli": "selfcheck_nli.SelfCheckNLIDetector",
        "selfcheck_bertscore": "selfcheck_bertscore.SelfCheckBERTScoreDetector",
        "semantic_entropy": "semantic_entropy.SemanticEntropyDetector",
        "lexical_similarity": "lexical_similarity.LexicalSimilarityDetector",
        "verbalize": "verbalize.VerbalizeDetector",
        "self_evaluator": "self_evaluator.SelfEvaluatorDetector",
        "perplexity": "uncertainty_metrics.PerplexityDetector",
        "ln_entropy": "uncertainty_metrics.LNEntropyDetector",
        "eigenscore_internal": "eigenscore.EigenScoreInternalDetector",
        "ccs": "ccs.CCSDetector",
        "prism": "prism.PRISMDetector",
        "saplma": "saplma.SAPLMADetector",
        "icr_probe": "icr_probe.ICRProbeDetector",
        "sep": "sep.SEPDetector",
        "mind": "mind.MINDDetector",
        "sar": "sar.SARDetector",
        "haloscope": "haloscope.HaloScopeDetector",
        "tsv": "tsv.TSVDetector"
    }
    if name not in detectors_map: raise ValueError(f"❌ Unknown target detector configuration: {name}")
    module_path, class_name = detectors_map[name].split('.')
    module = __import__(f"detectors.{module_path}", fromlist=[class_name])
    return getattr(module, class_name)(name=name)


def build_accessors(jsonl_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict=None):
    accessors = []
    if not os.path.exists(jsonl_path): return accessors

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            meta = json.loads(line)
            sid = str(meta["sample_id"])

            if aux_dict and sid in aux_dict:
                if aux_dict[sid].get("verbalize_response") is not None:
                    meta["verbalize_response"] = aux_dict[sid]["verbalize_response"]
                if aux_dict[sid].get("self_evaluator_raw") is not None:
                    meta["self_evaluator_raw"] = aux_dict[sid]["self_evaluator_raw"]
                    meta["self_evaluator_response"] = aux_dict[sid]["self_evaluator_raw"]

            acc = SampleAccessor(
                sample_id=sid, metadata=meta,
                h5_group=base_h5.get(sid) if base_h5 else None,
                stochastic_samples_dict=st_dict,
                stochastic_h5_group=st_h5.get(sid) if st_h5 else None
            )
            acc.recovered_logprobs = recovery_dict.get(sid)
            accessors.append(acc)
    return accessors


def run_benchmark(target_models: list, datasets: list, baselines: list):
    active_detectors = [get_detector(name) for name in baselines]

    need_stochastic = any(getattr(d, "requires_stochastic", False) for d in active_detectors)
    need_stochastic_hs = any(getattr(d, "requires_stochastic_hidden_states", False) for d in active_detectors)
    need_aux_eval = any(name in ["verbalize", "self_evaluator"] for name in baselines)
    need_logprobs = any(getattr(d, "requires_logprobs", False) for d in active_detectors)

    for target_model in target_models:
        for dataset_name in datasets:
            print(
                f"\n" + "=" * 60 + f"\n🚀 Initializing Strict Evaluation Pipeline: {target_model} | {dataset_name}\n" + "=" * 60)

            exp_dir = os.path.join(project_root, "experiments", target_model, f"{dataset_name}_10000samples")
            os.makedirs(exp_dir, exist_ok=True)

            base_meta_path = os.path.join(exp_dir, "03_final_scored_metadata.jsonl")
            train_meta_path = os.path.join(exp_dir, "03_train.jsonl")
            val_meta_path = os.path.join(exp_dir, "03_val.jsonl")
            test_meta_path = os.path.join(exp_dir, "03_test.jsonl")

            if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path) or not os.path.exists(
                    test_meta_path):
                print(f"\n[*] Executing Train/Val/Test data partitioning automatically...")
                split_dataset(input_jsonl=base_meta_path, exp_dir=exp_dir)

            st_jsonl_path = os.path.join(exp_dir, "04_stochastic_samples.jsonl")
            st_h5_path = os.path.join(exp_dir, "04_stochastic_hidden_states.h5")
            aux_jsonl_path = os.path.join(exp_dir, "04_auxiliary_evals.jsonl")
            base_h5_path = os.path.join(exp_dir, "02_hidden_states.h5")
            recovery_h5_path = os.path.join(exp_dir, "05_qa_features_base_logit_recovery.h5")
            final_report_path = os.path.join(exp_dir, "06_evaluation_results.json")

            with open(base_meta_path, 'r', encoding='utf-8') as f_meta:
                expected_total = sum(1 for _ in f_meta)

            sdpa_model, sdpa_tokenizer = None, None
            eager_model, eager_tokenizer = None, None

            def get_sdpa_model():
                nonlocal sdpa_model, sdpa_tokenizer
                if sdpa_model is None:
                    print(
                        f"\n[Commander] Initializing Fast Phase: Loading global model parameters via SDPA configuration -> {target_model}")
                    sdpa_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
                    if sdpa_tokenizer.pad_token is None: sdpa_tokenizer.pad_token = sdpa_tokenizer.eos_token
                    kwargs = EVAL_CONFIG["model_kwargs"].copy()
                    kwargs["attn_implementation"] = "sdpa"
                    sdpa_model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto",
                                                                      torch_dtype=torch.bfloat16, **kwargs).eval()
                return sdpa_model, sdpa_tokenizer

            def get_eager_model():
                nonlocal eager_model, eager_tokenizer
                if eager_model is None:
                    print(
                        f"\n[Commander] Initializing Intensive Analysis Phase: Loading global model parameters via Eager configuration -> {target_model}")
                    eager_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
                    if eager_tokenizer.pad_token is None: eager_tokenizer.pad_token = eager_tokenizer.eos_token
                    kwargs = EVAL_CONFIG["model_kwargs"].copy()
                    kwargs["attn_implementation"] = "eager"
                    eager_model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto",
                                                                       torch_dtype=torch.bfloat16, **kwargs).eval()
                return eager_model, eager_tokenizer

            if need_stochastic:
                print(f"[*] Executing Phase 1 stochastic sampling cache and structural resume check...")
                if enforce_safe_resume(base_meta_path, st_jsonl_path, st_h5_path if need_stochastic_hs else None,
                                       expected_total):
                    m, t = get_sdpa_model()
                    extractor = StochasticExtractor(model_name=target_model, model=m, tokenizer=t,
                                                    model_kwargs=EVAL_CONFIG["model_kwargs"])
                    extractor.process_stochastic_from_file(
                        input_jsonl_path=base_meta_path,
                        output_h5_path=st_h5_path, output_jsonl_path=st_jsonl_path,
                        layer_config=EVAL_CONFIG["layer_config"], token_config=EVAL_CONFIG["token_config"],
                        max_new_tokens=EVAL_CONFIG["max_new_tokens"], num_samples=EVAL_CONFIG["num_samples"],
                        system_prompt=EVAL_CONFIG["system_prompt"], num_shots=EVAL_CONFIG["num_shots"],
                        generation_kwargs=EVAL_CONFIG["stochastic_gen_kwargs"],
                        template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=False, run_self_evaluator=False, extract_stochastic_hs=need_stochastic_hs
                    )
                    del extractor
                    torch.cuda.empty_cache()

            if need_aux_eval:
                print(f"[*] Executing Phase 1.2 auxiliary metrics tracking and checkpoint verify...")
                if enforce_safe_resume(base_meta_path, aux_jsonl_path, None, expected_total):
                    m, t = get_sdpa_model()
                    aux_evaluator = AuxiliaryEvaluator(model_name=target_model, model=m, tokenizer=t,
                                                       model_kwargs=EVAL_CONFIG["model_kwargs"])
                    aux_evaluator.process_auxiliary_from_file(
                        input_jsonl_path=base_meta_path,
                        output_jsonl_path=aux_jsonl_path,
                        system_prompt=EVAL_CONFIG["system_prompt"], num_shots=EVAL_CONFIG["num_shots"],
                        template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=("verbalize" in baselines), run_self_evaluator=("self_evaluator" in baselines),
                    )
                    del aux_evaluator
                    torch.cuda.empty_cache()

            if need_logprobs:
                print(f"\n[*] Scanning file status configurations for 'base_logit_recovery' updates...")
                if enforce_safe_resume(base_meta_path, None, recovery_h5_path, expected_total):
                    m, t = get_sdpa_model()
                    process_dataset(input_jsonl=base_meta_path, output_h5=recovery_h5_path, model_name=target_model,
                                    method="base_logit_recovery", model_kwargs=EVAL_CONFIG["model_kwargs"], model=m,
                                    tokenizer=t, pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                    torch.cuda.empty_cache()

            for det in active_detectors:
                if getattr(det, "requires_qa_features", False) and det.name not in ["saplma", "icr_probe", "haloscope",
                                                                                    "mind", "tsv"]:
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    print(f"\n[*] Evaluating execution metrics files for isolated feature processing: {det.name}...")
                    if enforce_safe_resume(base_meta_path, None, qa_path, expected_total):
                        m, t = get_sdpa_model()
                        process_dataset(input_jsonl=base_meta_path, output_h5=qa_path, model_name=target_model,
                                        method=det.name, model_kwargs=EVAL_CONFIG["model_kwargs"], model=m, tokenizer=t,
                                        pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                        torch.cuda.empty_cache()

            if "tsv" in baselines:
                tsv_path = os.path.join(exp_dir, "05_qa_features_tsv.jsonl")

                if not os.path.exists(tsv_path):
                    print(f"\n[*] Launching full TSV supervised processing loop (Train -> Val Sweep -> Test Insert)...")
                    m, t = get_sdpa_model()
                    from baseline_detectors.data_utils.extract_tsv_features import TSVFeatureExtractor
                    tsv_extractor = TSVFeatureExtractor(model_name=target_model, model=m, tokenizer=t)

                    sweep_layers = [9, 11, 13, 15, 17, 19]
                    best_layer = -1
                    best_val_auroc = 0.0

                    best_trained_vector = None
                    best_centroids = None

                    val_labels = {}
                    with open(val_meta_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            d = json.loads(line)
                            if d.get("eval_category") in ["correct", "hallucination"]:
                                val_labels[str(d["sample_id"])] = 1 if d["eval_category"] == "hallucination" else 0

                    from sklearn.metrics import roc_auc_score

                    for current_layer in sweep_layers:
                        trained_vec, centroids = tsv_extractor.train_vector(
                            train_jsonl_path=train_meta_path,
                            str_layer=current_layer
                        )

                        temp_val_tsv = os.path.join(exp_dir, f"temp_val_layer{current_layer}.jsonl")
                        tsv_extractor.evaluate_vector(
                            eval_jsonl_path=val_meta_path,
                            output_jsonl_path=temp_val_tsv,
                            trained_vector=trained_vec,
                            final_centroids=centroids,
                            str_layer=current_layer
                        )

                        y_true, y_pred = [], []
                        with open(temp_val_tsv, 'r', encoding='utf-8') as f_val:
                            for line in f_val:
                                d = json.loads(line)
                                sid = str(d.get("sample_id"))
                                score = d.get("tsv_hallucination_score")
                                if sid in val_labels and score is not None:
                                    y_true.append(val_labels[sid])
                                    y_pred.append(float(score))

                        if len(set(y_true)) > 1:
                            val_auroc = roc_auc_score(y_true, y_pred)
                            print(f"    👉 [Sweep Progress] Layer {current_layer} | Val AUROC: {val_auroc * 100:.2f}%")

                            if val_auroc > best_val_auroc:
                                best_val_auroc = val_auroc
                                best_layer = current_layer
                                best_trained_vector = trained_vec.clone()
                                best_centroids = centroids.clone()

                        if os.path.exists(temp_val_tsv): os.remove(temp_val_tsv)

                    print(
                        f"🏆 [Sweep Finished] Identified optimal layer parameters: Layer {best_layer} (Val AUROC: {best_val_auroc * 100:.2f}%)")

                    print(
                        f"\n[*] Inserting identified optimal weights to target model layer {best_layer}. Deploying inference over Test set configuration...")

                    tsv_extractor.evaluate_vector(
                        eval_jsonl_path=test_meta_path,
                        output_jsonl_path=tsv_path,
                        trained_vector=best_trained_vector,
                        final_centroids=best_centroids,
                        str_layer=best_layer
                    )

                    del tsv_extractor
                    torch.cuda.empty_cache()
                else:
                    print(f"[+] Detected existing TSV structural tracking records. Skipping execution pass.")

            if sdpa_model is not None:
                del sdpa_model, sdpa_tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            for det in active_detectors:
                if getattr(det, "requires_qa_features", False) and det.name == "icr_probe":
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    print(f"\n[*] Verifying computational trace for Eager routine model extraction: {det.name}...")
                    if enforce_safe_resume(base_meta_path, None, qa_path, expected_total):
                        m, t = get_eager_model()
                        process_dataset(input_jsonl=base_meta_path, output_h5=qa_path, model_name=target_model,
                                        method=det.name, model_kwargs=EVAL_CONFIG["model_kwargs"], model=m, tokenizer=t,
                                        pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                        torch.cuda.empty_cache()

            if eager_model is not None:
                del eager_model, eager_tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            print(f"\n[*] Loading offline precomputed cache metrics and array matrices...")
            st_dict = {}
            if os.path.exists(st_jsonl_path):
                with open(st_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            st_dict[str(d["sample_id"])] = {"samples": d.get("stochastic_samples", []),
                                                            "log_likelihoods": d.get("stochastic_log_likelihoods", [])}

            aux_dict = {}
            if os.path.exists(aux_jsonl_path):
                with open(aux_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            aux_dict[str(d["sample_id"])] = {"verbalize_response": d.get("verbalize_response"),
                                                             "self_evaluator_raw": d.get("self_evaluator_raw")}

            recovery_dict = {}
            if os.path.exists(recovery_h5_path):
                with h5py.File(recovery_h5_path, 'r') as rec_h5:
                    for k in rec_h5.keys(): recovery_dict[k.replace("_base_logit_recovery", "")] = rec_h5[k][
                                                                                                       "logprobs"][:]

            base_h5 = h5py.File(base_h5_path, 'r') if os.path.exists(base_h5_path) else None
            st_h5 = h5py.File(st_h5_path, 'r') if os.path.exists(st_h5_path) else None

            qa_h5_handles = {}
            for det in active_detectors:
                if getattr(det, "requires_qa_features", False):
                    if det.name in ["saplma", "haloscope"]:
                        qa_h5_handles[det.name] = h5py.File(recovery_h5_path, 'r')
                    elif det.name == "mind":
                        continue
                    else:
                        qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                        if os.path.exists(qa_path): qa_h5_handles[det.name] = h5py.File(qa_path, 'r')

            train_accessors = build_accessors(train_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)
            test_accessors = build_accessors(test_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)

            print(
                f"[*] Structural cross-split environment isolation verified. Train: {len(train_accessors)} | Test: {len(test_accessors)}")

            raw_scores = {acc.sample_id: {} for acc in test_accessors}
            for detector in active_detectors:
                print(f"\n>>> Running method pipeline: {detector.name}...")
                if getattr(detector, "requires_qa_features", False) and detector.name != "mind":
                    target_h5 = qa_h5_handles.get(detector.name)
                    for acc in train_accessors + test_accessors:
                        acc.qa_h5_file = target_h5
                        acc.method_type = "base_logit_recovery" if detector.name in ["saplma",
                                                                                     "haloscope"] else detector.name

                detector.fit(train_accessors)
                for acc in tqdm(test_accessors, desc="Scoring", leave=False):
                    raw_scores[acc.sample_id][detector.name] = detector.predict_score(acc)

            final_report = {"metrics": {}, "raw_scores": raw_scores}
            for det in active_detectors:
                y_true, y_pred = [], []
                for acc in test_accessors:
                    label = acc.metadata.get("eval_category")
                    if label not in ["correct", "hallucination"]: continue
                    score = raw_scores[acc.sample_id][det.name]
                    if score is None or math.isnan(float(score)): continue
                    y_true.append(1 if label == "hallucination" else 0)
                    y_pred.append(float(score))

                if y_true and len(set(y_true)) > 1:
                    m = ClassificationEvaluator.compute_metrics(y_true, y_pred)
                    final_report["metrics"][det.name] = m
                    print(f"  - [{det.name:20}] AUROC: {m['AUROC']:.2f}%")

            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2)

            if base_h5: base_h5.close()
            if st_h5: st_h5.close()
            for h in qa_h5_handles.values(): h.close()


DEFAULT_MODELS = ["Qwen/Qwen3-14B"]
DEFAULT_DATASETS = ["belebele"]
DEFAULT_BASELINES = [
    "selfcheck_bertscore",
    "selfcheck_nli",
    "semantic_entropy",
    "lexical_similarity",
    "verbalize",
    "self_evaluator",
    "perplexity",
    "ln_entropy",
    "eigenscore_internal",
    "ccs",
    "prism",
    "saplma",
    "sep",
    "icr_probe",
    "mind",
    "sar",
    "haloscope",
    "tsv",
]


def parse_args():
    """Parse command-line overrides. With no arguments, the defaults below
    reproduce the original hard-coded run configuration."""
    import argparse
    p = argparse.ArgumentParser(
        description="OpenHalDet — Phase 2: detector evaluation. "
                    "Select backbones, datasets, and detectors to benchmark under a unified protocol."
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="One or more backbone LLM ids (space-separated).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="One or more dataset keys (space-separated).")
    p.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES,
                   help="One or more detector registry keys (space-separated). "
                        "Omit to run all supported detectors.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)
    run_benchmark(
        target_models=args.models,
        datasets=args.datasets,
        baselines=args.baselines,
    )