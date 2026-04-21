import os
import sys
import json
import h5py
import torch
import math
from tqdm import tqdm

# ==========================================
# 🎯 环境注入：挂载数据模块 (完美修复路径迷失)
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
# 💎 评测全局配置中枢
# ==========================================
EVAL_CONFIG = {
    "num_samples": 5,
    "max_new_tokens": 2048,
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 4, 
    "layer_config": {"mode": "middle", "count": 5},
    "token_config": {"mode": "backward", "count": 5},
    "stochastic_gen_kwargs": {"num_beams": 5, "do_sample": True, "temperature": 1.0},
    "template_kwargs": {"enable_thinking": False},
    "model_kwargs": {"trust_remote_code": True, "attn_implementation": "sdpa"}
}

def validate_stochastic_cache(jsonl_path: str, active_detectors: list, expected_total: int) -> bool:
    """鲁棒性特征校验器：深度检查 JSONL 缓存文件的完整性。"""
    if not os.path.exists(jsonl_path):
        return False
        
    required_keys = ["sample_id"]
    if any(getattr(d, "requires_stochastic", False) for d in active_detectors):
        required_keys.append("stochastic_samples")
        
    try:
        valid_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if not all(k in data for k in required_keys):
                    print(f"[-] 缓存失效: 样本 {data.get('sample_id')} 缺失关键字段，当前要求包含 {required_keys}")
                    return False
                if "stochastic_samples" in required_keys and not data.get("stochastic_samples"):
                    print(f"[-] 缓存失效: 样本 {data.get('sample_id')} 的 stochastic_samples 为空")
                    return False
                valid_count += 1
                
        if valid_count < expected_total:
            print(f"[-] 缓存失效: 样本数量不匹配 (当前缓存 {valid_count} 个，预期需要 {expected_total} 个)")
            return False
            
        return True
    except Exception as e:
        print(f"[-] 缓存文件损坏或解析异常: {e}")
        return False

def validate_aux_cache(jsonl_path: str, active_detectors: list, expected_total: int) -> bool:
    """校验 auxiliary eval 缓存文件完整性。"""
    if not os.path.exists(jsonl_path):
        return False

    required_keys = ["sample_id"]
    if any(d.name == "verbalize" for d in active_detectors):
        required_keys.append("verbalize_response")
    if any(d.name == "self_evaluator" for d in active_detectors):
        required_keys.append("self_evaluator_raw")

    try:
        valid_count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if not all(k in data for k in required_keys):
                    print(f"[-] Auxiliary 缓存失效: 样本 {data.get('sample_id')} 缺失关键字段，当前要求包含 {required_keys}")
                    return False
                valid_count += 1

        if valid_count < expected_total:
            print(f"[-] Auxiliary 缓存失效: 样本数量不匹配 (当前缓存 {valid_count} 个，预期需要 {expected_total} 个)")
            return False

        return True
    except Exception as e:
        print(f"[-] Auxiliary 缓存文件损坏或解析异常: {e}")
        return False

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
            
            # 🛠️ [修改位置]: verbalize / self_evaluator 改为从独立 aux_dict 挂载到 metadata
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
                print(f"\n[*] 检测到缺失数据集切分文件，大管家正在自动执行 Train/Val/Test 严格切分...")
                split_dataset(input_jsonl=base_meta_path, exp_dir=exp_dir)
                print(f"[+] 数据集切分完成！\n")

            st_jsonl_path = os.path.join(exp_dir, "04_stochastic_samples.jsonl")
            st_h5_path = os.path.join(exp_dir, "04_stochastic_hidden_states.h5")
            aux_jsonl_path = os.path.join(exp_dir, "04_auxiliary_evals.jsonl")
            base_h5_path = os.path.join(exp_dir, "02_hidden_states.h5")
            recovery_h5_path = os.path.join(exp_dir, "05_qa_features_base_logit_recovery.h5") 
            final_report_path = os.path.join(exp_dir, "06_evaluation_results.json")

            with open(base_meta_path, 'r', encoding='utf-8') as f_meta:
                expected_total = sum(1 for _ in f_meta)

            # 【阶段 1】采样引擎（只服务真正需要多次采样的方法）
            if need_stochastic:
                print(f"[*] 执行阶段 1 缓存完整性校验...")
                is_cache_valid = validate_stochastic_cache(st_jsonl_path, active_detectors, expected_total)
                
                if not is_cache_valid:
                    print(f"[!] 校验未通过：即将启动高级采样引擎进行全量重构...")
                    for dirty_file in [st_jsonl_path, st_h5_path]:
                        if os.path.exists(dirty_file):
                            os.remove(dirty_file)
                            print(f"  - 已清理历史脏数据: {os.path.basename(dirty_file)}")

                    extractor = StochasticExtractor(model_name=target_model, model_kwargs=EVAL_CONFIG["model_kwargs"])
                    extractor.process_stochastic_from_file(
                        input_jsonl_path=base_meta_path, output_h5_path=st_h5_path, output_jsonl_path=st_jsonl_path,
                        layer_config=EVAL_CONFIG["layer_config"], token_config=EVAL_CONFIG["token_config"],
                        max_new_tokens=EVAL_CONFIG["max_new_tokens"], num_samples=EVAL_CONFIG["num_samples"],
                        system_prompt=EVAL_CONFIG["system_prompt"], num_shots=EVAL_CONFIG["num_shots"],
                        generation_kwargs=EVAL_CONFIG["stochastic_gen_kwargs"], template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=False, run_self_evaluator=False,
                        extract_stochastic_hs=need_stochastic_hs
                    )
                    del extractor
                    torch.cuda.empty_cache()
                else:
                    print(f"[+] 采样缓存校验通过！完全满足当前 {len(active_detectors)} 个探测器的严苛要求，直接复用。")

            # 【阶段 1.2】auxiliary eval 引擎（verbalize / self_evaluator 独立生成）
            if need_aux_eval:
                print(f"[*] 执行阶段 1.2 Auxiliary 缓存完整性校验...")
                is_aux_cache_valid = validate_aux_cache(aux_jsonl_path, active_detectors, expected_total)

                if not is_aux_cache_valid:
                    print(f"[!] Auxiliary 校验未通过：即将启动独立生成引擎...")
                    if os.path.exists(aux_jsonl_path):
                        os.remove(aux_jsonl_path)
                        print(f"  - 已清理历史脏数据: {os.path.basename(aux_jsonl_path)}")

                    aux_evaluator = AuxiliaryEvaluator(model_name=target_model, model_kwargs=EVAL_CONFIG["model_kwargs"])
                    aux_evaluator.process_auxiliary_from_file(
                        input_jsonl_path=base_meta_path,
                        output_jsonl_path=aux_jsonl_path,
                        system_prompt=EVAL_CONFIG["system_prompt"],
                        num_shots=EVAL_CONFIG["num_shots"],
                        template_kwargs=EVAL_CONFIG["template_kwargs"],
                        run_verbalize=("verbalize" in baselines),
                        run_self_evaluator=("self_evaluator" in baselines),
                    )
                    del aux_evaluator
                    torch.cuda.empty_cache()
                else:
                    print(f"[+] Auxiliary 缓存校验通过！直接复用。")

            # 【阶段 1.5】全自动造子弹
            if need_logprobs and not os.path.exists(recovery_h5_path):
                print(f"\n[*] 启动 'base_logit_recovery' 独立补票...")
                process_dataset(input_jsonl=base_meta_path, output_h5=recovery_h5_path, model_name=target_model, method="base_logit_recovery", model_kwargs=EVAL_CONFIG["model_kwargs"])
                torch.cuda.empty_cache()

            for det in active_detectors:
                if getattr(det, "requires_qa_features", False) and det.name != "saplma":
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    
                    need_extract = False
                    if not os.path.exists(qa_path):
                        need_extract = True
                    else:
                        try:
                            with h5py.File(qa_path, 'r') as f_check:
                                all_keys = list(f_check.keys())
                                if not all_keys or len(all_keys) < expected_total:
                                    print(f"[*] 发现 {det.name} 特征不完整 (当前 {len(all_keys)} 个，预期 {expected_total} 个)！准备重跑...")
                                    need_extract = True
                                
                                else:
                                    sample_grp = f_check[all_keys[0]]
                                    if det.name == "self_evaluator" and "logprobs" not in sample_grp:
                                        print(f"[*] 检测到 {det.name} 缺失对数概率 (logprobs)，准备重跑...")
                                        need_extract = True
                                    elif det.name == "icr_probe" and "icr_feature" not in sample_grp:
                                        print(f"[*] 检测到 {det.name} 核心字段缺失，准备重跑...")
                                        need_extract = True
                                    elif det.name == "sep" and "sep_points" not in sample_grp:
                                        print(f"[*] 检测到 {det.name} 核心字段缺失 (sep_points)，准备重跑...")
                                        need_extract = True
                                
                        except Exception as e:
                            print(f"[*] 发现 {det.name} 特征文件已损坏，准备重跑提取...")
                            need_extract = True

                    if need_extract:
                        print(f"\n[*] 启动 {det.name} 专属特征提取引擎...")
                        process_dataset(input_jsonl=base_meta_path, output_h5=qa_path, model_name=target_model, method=det.name, model_kwargs=EVAL_CONFIG["model_kwargs"])
                        torch.cuda.empty_cache()

            # 【阶段 2】加载特征
            print(f"[*] 加载数据特征 (建立 Train / Test 屏障)...")
            st_dict = {}
            if os.path.exists(st_jsonl_path):
                with open(st_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        d = json.loads(line)
                        st_dict[str(d["sample_id"])] = {
                            "samples": d.get("stochastic_samples", []), 
                            "log_likelihoods": d.get("stochastic_log_likelihoods", [])
                        }

            aux_dict = {}
            if os.path.exists(aux_jsonl_path):
                with open(aux_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        d = json.loads(line)
                        aux_dict[str(d["sample_id"])] = {
                            "verbalize_response": d.get("verbalize_response"),
                            "self_evaluator_raw": d.get("self_evaluator_raw")
                        }

            recovery_dict = {}
            if os.path.exists(recovery_h5_path):
                with h5py.File(recovery_h5_path, 'r') as rec_h5:
                    for group_name in rec_h5.keys():
                        recovery_dict[group_name.replace("_base_logit_recovery", "")] = rec_h5[group_name]["logprobs"][:]

            base_h5 = h5py.File(base_h5_path, 'r') if os.path.exists(base_h5_path) else None
            st_h5 = h5py.File(st_h5_path, 'r') if os.path.exists(st_h5_path) else None

            qa_h5_handles = {}
            for det in active_detectors:
                if getattr(det, "requires_qa_features", False):
                    if det.name == "saplma":
                        qa_h5_handles["saplma"] = h5py.File(recovery_h5_path, 'r')
                        continue
                    qa_path = os.path.join(exp_dir, f"05_qa_features_{det.name}.h5")
                    if os.path.exists(qa_path): qa_h5_handles[det.name] = h5py.File(qa_path, 'r')

            # 🚀 按花名册提人
            train_accessors = build_accessors(train_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)
            test_accessors = build_accessors(test_meta_path, base_h5, st_dict, st_h5, recovery_dict, aux_dict)

            print(f"[*] 数据隔离完毕。Train: {len(train_accessors)} 个 | Test: {len(test_accessors)} 个")

            # 【阶段 3】打分流程
            raw_scores = {acc.sample_id: {} for acc in test_accessors}
            for detector in active_detectors:
                print(f"\n>>> {detector.name} 运行中...")
                if getattr(detector, "requires_qa_features", False):
                    target_h5 = qa_h5_handles.get(detector.name)
                    for acc in train_accessors + test_accessors:
                        acc.qa_h5_file = target_h5
                        acc.method_type = detector.name 
                
                detector.fit(train_accessors)
                for acc in tqdm(test_accessors, desc="Scoring on Test Set", leave=False):
                    raw_scores[acc.sample_id][detector.name] = detector.predict_score(acc)

            # 【阶段 4】最终算分
            print(f"\n[*] 写入实验报告 (仅限测试集)...")
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
                    print(f"  - [{det.name:20}] AUROC: {m['AUROC']:.2f}% (测试样本: {len(y_true)})")

            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)

            if base_h5: base_h5.close()
            if st_h5: st_h5.close()
            for h in qa_h5_handles.values():
                if h: h.close()

if __name__ == "__main__":
    run_benchmark(
        target_models=["meta-llama/Llama-3.1-8B-Instruct"],
        datasets=[
            # "truthful_qa",
            # "halueval_qa",
            # "trivia_qa",
            # "coqa",
            # "squad_v2",
            # "arc_challenge",
            # "xsum",
            # "gsm8k",
            # "human_eval",
            # "xlam_agent",
            # "mbpp"
        ],
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
            "sar"
        ] 
    )