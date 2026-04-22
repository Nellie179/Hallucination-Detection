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
    """全局随机种子固定：保证每次采样与特征提取绝对一致"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] 全局随机种子已牢牢锁死为: {seed}")

# ==========================================
# 🎯 环境注入：挂载数据模块
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "datasets_v1")
utils_dir = os.path.join(project_root, "data_utils")

if project_root not in sys.path: sys.path.insert(0, project_root)
if data_dir not in sys.path: sys.path.insert(0, data_dir)
if utils_dir not in sys.path: sys.path.insert(0, utils_dir)

# 纯净导入
from datasets_v1.generate_stochastic_samples import StochasticExtractor
from datasets_v1.generate_auxiliary_evals import AuxiliaryEvaluator
from data_utils.accessor import SampleAccessor
from baseline_detectors.evaluators.classification import ClassificationEvaluator
from data_utils.extract_qa_hidden_states import process_dataset
from data_utils.data_split import split_dataset 

# ==========================================
# ⚙️ 采样策略引擎 (双擎版)
# ==========================================
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
    "pooling_method": "mean", # 可选 "mean" 或 "last"
    "template_kwargs": {"enable_thinking": False}, 
    "model_kwargs": {"trust_remote_code": True}
}

# =======================================================================================
# 🚀 军工级核心：退二保底无痕容灾器 (Clean Safe Resume) - 兼容死信队列(DLQ)版
# =======================================================================================
def enforce_safe_resume(base_meta_path, target_jsonl, target_h5, expected_total, rollback_count=2):
    """
    检查进度并切除可能损坏的脏尾巴。
    智能区分：
    1. 中途断电 -> 执行 Rollback 物理切除脏尾巴
    2. 留洞跑完 (OOM) -> 进入无损补洞模式，不删数据
    """
    ordered_items = []
    with open(base_meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): ordered_items.append(json.loads(line))

    # 1. 扫描已生成的 ID
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
                except: break

    h5_ids = set()
    if target_h5 and os.path.exists(target_h5):
        try:
            with h5py.File(target_h5, 'r') as f_h5:
                h5_keys = set(f_h5.keys())
                for item in ordered_items:
                    sid = str(item["sample_id"])
                    if sid in h5_keys or any(k.startswith(sid + "_") for k in h5_keys):
                        h5_ids.add(sid)
        except Exception: pass

    # 2. 核心补丁：查阅死信队列 (DLQ)，将确诊死亡的也算作进度
    dlq_file = target_h5 + ".failed_ids.txt" if target_h5 else (target_jsonl + ".failed_ids.txt" if target_jsonl else None)
    failed_ids = set()
    if dlq_file and os.path.exists(dlq_file):
        with open(dlq_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    failed_ids.add(line.split('\t')[0].strip())
    
    # 进度交集 (谁慢听谁的)
    if target_jsonl and target_h5: processed_ids = json_ids.intersection(h5_ids)
    elif target_jsonl: processed_ids = json_ids
    elif target_h5: processed_ids = h5_ids
    else: processed_ids = set()

    # 有效总进度 = 成功录入 + 确诊死亡
    effective_ids = processed_ids.union(failed_ids)

    if len(effective_ids) == 0: return True 

    # 找到物理遍历达到的最高索引
    highest_idx = -1
    for i, item in enumerate(ordered_items):
        if str(item["sample_id"]) in effective_ids:
            highest_idx = i

    # 情况 A：100% 跑完 (成功 + 失败 = 预期总数)
    if len(effective_ids) >= expected_total and highest_idx == expected_total - 1:
        if failed_ids:
            print(f"    [+] 进度 100% (其中 {len(failed_ids)} 条在死信队列中)，无需重复运行。")
        return False 

    # 情况 B：虽然总数不够，但已经跑到了全量试卷的最后一条 (留洞模式)
    if highest_idx == expected_total - 1:
        missing = expected_total - len(effective_ids)
        print(f"    🛠️ 发现进度已达末尾，但存在 {missing} 个样本空洞。触发无损补洞模式...")
        return True # 直接下发原卷，让底层去补坑，绝对不执行 Rollback！

    # 情况 C：中途物理断电 (Highest Index 未达末尾)
    safe_limit_idx = max(-1, highest_idx - rollback_count)
    safe_ids = set([str(item["sample_id"]) for item in ordered_items[:safe_limit_idx + 1]])

    print(f"    🛠️ 发现进度中途物理中断 (断点: {highest_idx + 1}/{expected_total})。执行退二清理 (-{rollback_count})...")

    if target_jsonl and os.path.exists(target_jsonl):
        with open(target_jsonl, 'w', encoding='utf-8') as f:
            for line in valid_json_lines:
                try:
                    if str(json.loads(line)["sample_id"]) in safe_ids: f.write(line)
                except: pass

    if target_h5 and os.path.exists(target_h5):
        try:
            with h5py.File(target_h5, 'a') as f_h5:
                for key in list(f_h5.keys()):
                    # 这里注意：failed_ids 也可以被清理，因为如果是断电，我们希望干净地重跑
                    if not any(key == sid or key.startswith(sid + "_") for sid in safe_ids):
                        del f_h5[key]
        except Exception: pass

    print(f"    [+] 尾部脏数据清理完毕！将由底层引擎自动跳过前 {len(safe_ids)} 条，进行无缝追加。")
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
        "haloscope": "haloscope.HaloScopeDetector"
    }
    if name not in detectors_map: raise ValueError(f"❌ 未知探测器: {name}")
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
            print(f"\n" + "="*60 + f"\n🚀 启动严格 Test 评测: {target_model} | {dataset_name}\n" + "="*60)
            
            exp_dir = os.path.join(project_root, "experiments", target_model, f"{dataset_name}_10000samples")
            os.makedirs(exp_dir, exist_ok=True)
            
            base_meta_path = os.path.join(exp_dir, "03_final_scored_metadata.jsonl")
            train_meta_path = os.path.join(exp_dir, "03_train.jsonl")
            test_meta_path = os.path.join(exp_dir, "03_test.jsonl")
            
            if not os.path.exists(train_meta_path) or not os.path.exists(test_meta_path):
                print(f"\n[*] 自动执行 Train/Val/Test 严格切分...")
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
                    print(f"\n[Commander] 启动极速突击相位：加载全局大模型 (SDPA 模式) -> {target_model}")
                    sdpa_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
                    if sdpa_tokenizer.pad_token is None: sdpa_tokenizer.pad_token = sdpa_tokenizer.eos_token
                    kwargs = EVAL_CONFIG["model_kwargs"].copy()
                    kwargs["attn_implementation"] = "sdpa"
                    sdpa_model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto", torch_dtype=torch.bfloat16, **kwargs).eval()
                return sdpa_model, sdpa_tokenizer

            def get_eager_model():
                nonlocal eager_model, eager_tokenizer
                if eager_model is None:
                    print(f"\n[Commander] 启动深度分析相位：加载全局大模型 (Eager 模式) -> {target_model}")
                    eager_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
                    if eager_tokenizer.pad_token is None: eager_tokenizer.pad_token = eager_tokenizer.eos_token
                    kwargs = EVAL_CONFIG["model_kwargs"].copy()
                    kwargs["attn_implementation"] = "eager"
                    eager_model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto", torch_dtype=torch.bfloat16, **kwargs).eval()
                return eager_model, eager_tokenizer

            # 【1.1】采样引擎
            if need_stochastic:
                print(f"[*] 执行阶段 1 采样引擎缓存与断点校验...")
                if enforce_safe_resume(base_meta_path, st_jsonl_path, st_h5_path if need_stochastic_hs else None, expected_total):
                    m, t = get_sdpa_model()
                    extractor = StochasticExtractor(model_name=target_model, model=m, tokenizer=t, model_kwargs=EVAL_CONFIG["model_kwargs"])
                    extractor.process_stochastic_from_file(
                        input_jsonl_path=base_meta_path, # 直接下发原卷
                        output_h5_path=st_h5_path, output_jsonl_path=st_jsonl_path,
                        layer_config=EVAL_CONFIG["layer_config"], token_config=EVAL_CONFIG["token_config"],
                        max_new_tokens=EVAL_CONFIG["max_new_tokens"], num_samples=EVAL_CONFIG["num_samples"],
                        system_prompt=EVAL_CONFIG["system_prompt"], num_shots=EVAL_CONFIG["num_shots"],
                        generation_kwargs=EVAL_CONFIG["stochastic_gen_kwargs"], template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=False, run_self_evaluator=False, extract_stochastic_hs=need_stochastic_hs
                    )
                    del extractor
                    torch.cuda.empty_cache()

            # 【1.2】Auxiliary Eval
            if need_aux_eval:
                print(f"[*] 执行阶段 1.2 Auxiliary 缓存与断点校验...")
                if enforce_safe_resume(base_meta_path, aux_jsonl_path, None, expected_total):
                    m, t = get_sdpa_model()
                    aux_evaluator = AuxiliaryEvaluator(model_name=target_model, model=m, tokenizer=t, model_kwargs=EVAL_CONFIG["model_kwargs"])
                    aux_evaluator.process_auxiliary_from_file(
                        input_jsonl_path=base_meta_path,
                        output_jsonl_path=aux_jsonl_path,
                        system_prompt=EVAL_CONFIG["system_prompt"], num_shots=EVAL_CONFIG["num_shots"],
                        template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=("verbalize" in baselines), run_self_evaluator=("self_evaluator" in baselines),
                    )
                    del aux_evaluator
                    torch.cuda.empty_cache()

            # 【1.5】Logit Recovery (SAPLMA/HaloScope 基础特征)
            if need_logprobs:
                print(f"\n[*] 扫描 'base_logit_recovery' 补票进度...")
                if enforce_safe_resume(base_meta_path, None, recovery_h5_path, expected_total):
                    m, t = get_sdpa_model()
                    process_dataset(input_jsonl=base_meta_path, output_h5=recovery_h5_path, model_name=target_model, method="base_logit_recovery", model_kwargs=EVAL_CONFIG["model_kwargs"], model=m, tokenizer=t, pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                    torch.cuda.empty_cache()

            # 【1.6】其他 SDPA 探测器
            for det in active_detectors:
                if getattr(det, "requires_qa_features", False) and det.name not in ["saplma", "icr_probe", "haloscope", "mind"]:
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    print(f"\n[*] 扫描 {det.name} 专属特征提取进度...")
                    if enforce_safe_resume(base_meta_path, None, qa_path, expected_total):
                        m, t = get_sdpa_model()
                        process_dataset(input_jsonl=base_meta_path, output_h5=qa_path, model_name=target_model, method=det.name, model_kwargs=EVAL_CONFIG["model_kwargs"], model=m, tokenizer=t, pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                        torch.cuda.empty_cache()

            if sdpa_model is not None:
                del sdpa_model, sdpa_tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            # 【相位 2】Eager 模式 (ICR Probe)
            for det in active_detectors:
                if getattr(det, "requires_qa_features", False) and det.name == "icr_probe":
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    print(f"\n[*] 扫描 {det.name} (Eager) 提取进度...")
                    if enforce_safe_resume(base_meta_path, None, qa_path, expected_total):
                        m, t = get_eager_model()
                        process_dataset(input_jsonl=base_meta_path, output_h5=qa_path, model_name=target_model, method=det.name, model_kwargs=EVAL_CONFIG["model_kwargs"], model=m, tokenizer=t, pooling=EVAL_CONFIG.get("pooling_method", "mean"))
                        torch.cuda.empty_cache()

            if eager_model is not None:
                del eager_model, eager_tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            # 【相位 3】纯离线评测
            print(f"\n[*] 加载离线特征中...")
            st_dict = {}
            if os.path.exists(st_jsonl_path):
                with open(st_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            st_dict[str(d["sample_id"])] = {"samples": d.get("stochastic_samples", []), "log_likelihoods": d.get("stochastic_log_likelihoods", [])}

            aux_dict = {}
            if os.path.exists(aux_jsonl_path):
                with open(aux_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            aux_dict[str(d["sample_id"])] = {"verbalize_response": d.get("verbalize_response"), "self_evaluator_raw": d.get("self_evaluator_raw")}

            recovery_dict = {}
            if os.path.exists(recovery_h5_path):
                with h5py.File(recovery_h5_path, 'r') as rec_h5:
                    for k in rec_h5.keys(): recovery_dict[k.replace("_base_logit_recovery", "")] = rec_h5[k]["logprobs"][:]

            base_h5 = h5py.File(base_h5_path, 'r') if os.path.exists(base_h5_path) else None
            st_h5 = h5py.File(st_h5_path, 'r') if os.path.exists(st_h5_path) else None

            qa_h5_handles = {}
            for det in active_detectors:
                if getattr(det, "requires_qa_features", False):
                    if det.name in ["saplma", "haloscope"]: qa_h5_handles[det.name] = h5py.File(recovery_h5_path, 'r')
                    elif det.name == "mind": continue
                    else:
                        qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                        if os.path.exists(qa_path): qa_h5_handles[det.name] = h5py.File(qa_path, 'r')

            train_accessors = build_accessors(train_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)
            test_accessors = build_accessors(test_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)

            print(f"[*] 数据隔离完毕。Train: {len(train_accessors)} | Test: {len(test_accessors)}")

            raw_scores = {acc.sample_id: {} for acc in test_accessors}
            for detector in active_detectors:
                print(f"\n>>> {detector.name} 运行中...")
                if getattr(detector, "requires_qa_features", False) and detector.name != "mind":
                    target_h5 = qa_h5_handles.get(detector.name)
                    for acc in train_accessors + test_accessors:
                        acc.qa_h5_file = target_h5
                        acc.method_type = "base_logit_recovery" if detector.name in ["saplma", "haloscope"] else detector.name
                
                detector.fit(train_accessors)
                for acc in tqdm(test_accessors, desc="Scoring", leave=False):
                    raw_scores[acc.sample_id][detector.name] = detector.predict_score(acc)

            # 报告输出
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

            with open(final_report_path, 'w', encoding='utf-8') as f: json.dump(final_report, f, indent=2)

            if base_h5: base_h5.close()
            if st_h5: st_h5.close()
            for h in qa_h5_handles.values(): h.close()

if __name__ == "__main__":
    set_global_seed(42)
    run_benchmark(
        target_models=["meta-llama/Llama-3.1-8B-Instruct"],
        datasets=["svamp"],
        baselines=[
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
            "haloscope"
        ] 
    )