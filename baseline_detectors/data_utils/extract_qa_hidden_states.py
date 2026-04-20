# /home/zfang1/Data/Lxy/Benchmark/baseline_detectors/data_utils/extract_qa_hidden_states.py
"""
Extract hidden states, LOGPROBS and ICR Scores from Q+A (Universal Version)
权威拼接版：基于 Tensor 物理拼接，确保 Token 绝对对齐并防止索引越界。
已新增 SEP (Semantic Entropy Probes) 所需的 TBG 和 SLT 特征提取逻辑。
"""

import os
import json
import torch
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🎯 引入 ICR 核心算法组件
from data_utils.icr_score import ICRScore

class QAHiddenStateExtractor:
    """支持全量张量提取与概率截获的终极提取器"""

    def __init__(
        self,
        model_name: str,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        model_kwargs: dict = None,
        method: str = None  # 👈 新增参数，用于判断是否需要切换 eager 模式
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_kwargs = (model_kwargs or {}).copy() # 👈 克隆一份，防止污染全局配置
        self.method = method

        # 🎯 核心修复：ICR Probe 必须使用 eager 模式才能吐出 Attention 矩阵
        if self.method == "icr_probe":
            if self.model_kwargs.get("attn_implementation") != "eager":
                print(f"[*] 注意：{self.method} 需要 Attention 矩阵，正在将实现切换为 eager...")
                self.model_kwargs["attn_implementation"] = "eager"

        print(f"[*] 正在加载模型 {model_name} 到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **self.model_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            **self.model_kwargs
        ).eval()

        if hasattr(self.model.config, "num_hidden_layers"):
            self.total_layers = self.model.config.num_hidden_layers
        else:
            self.total_layers = self.model.config.text_config.num_hidden_layers
            
        print(f"[+] 模型加载完毕。总层数: {self.total_layers}")

    def _pack_for_icr(self, hidden_states, attentions, prompt_len):
        """将 HF 输出转换为 ICRScore 结构，使用动态长度感应，防止任何层数不匹配"""
        # 🎯 增加一个安全检查：如果 attentions 为空，直接报错提醒环境问题
        if attentions is None or len(attentions) == 0:
            raise ValueError("未能获取到 Attention 矩阵。请确认 attn_implementation 为 'eager'。")

        # 🎯 直接转换为局部 tuple，确保每一行代码都只针对当前的真实数据
        hs_tuple = tuple(h.detach() for h in hidden_states)
        att_tuple = tuple(a.detach() for a in attentions)
        
        # 🛡️ 护盾 1：动态获取当前样本的真实层数
        actual_hs_layers = len(hs_tuple)
        actual_att_layers = len(att_tuple)
        
        # 获取序列总长度
        B, S, D = hs_tuple[0].shape
        output_size = max(0, S - prompt_len)

        nested_hs = []
        # 处理输入段 (Prompt)
        if actual_hs_layers > 0:
            # 🛡️ 护盾 2：这里 range 改用实际拿到的 actual_hs_layers
            hs_input = [hs_tuple[ell][0, :prompt_len, :].unsqueeze(0) for ell in range(actual_hs_layers)]
            nested_hs.append(hs_input)
        
        # 处理回答段 (逐 token)
        for t in range(output_size):
            pos = prompt_len + t
            if pos >= S: break # 🛡️ 护盾 3：物理越界拦截
            
            # 🛡️ 护盾 4：动态索引。如果某一层没拿到，列表生成式会直接报错，这里我们用 actual 保证安全
            token_hs = [hs_tuple[ell][0, pos:pos+1, :].unsqueeze(0) for ell in range(actual_hs_layers)]
            nested_hs.append(token_hs)

        nested_att = []
        # 处理输入段
        if actual_att_layers > 0:
            att_input = [att_tuple[ell][0].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(att_input)
        
        # 处理回答段 (逐 token)
        for t in range(output_size):
            pos = prompt_len + t
            if pos >= S: break
            
            # 🛡️ 护盾 5：针对 Attention 层数做独立的动态迭代
            token_att = [att_tuple[ell][0, :, pos:pos+1, :].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(token_att)
            
        return nested_hs, nested_att

    def extract_features(
        self,
        prompt: str,
        answer: str,
        method: str,
        extract_all_layers: bool = True
    ) -> Dict[str, Any]:
        # 1. 🎯 权威编码：分别 Encode 避免分词污染
        p_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        a_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # ==========================================
        # 🛡️ 战时截断补丁：防 Eager 模式 OOM
        # ==========================================
        MAX_SEQ_LEN = 128  # A100 剩余 33GB 时的绝对安全线
        a_len = a_ids.input_ids.shape[1]
        p_len = p_ids.input_ids.shape[1]
        
        # 只针对吃显存的 icr_probe 触发截断，且总长超标时才动手
        if method == "icr_probe" and (p_len + a_len) > MAX_SEQ_LEN:
            # 优先保证大模型的 Answer 完整，截断 Prompt 的头部 (保留离答案最近的上下文)
            keep_p_len = max(10, MAX_SEQ_LEN - a_len) 
            p_ids.input_ids = p_ids.input_ids[:, -keep_p_len:]
            p_ids.attention_mask = p_ids.attention_mask[:, -keep_p_len:]
            print(f"  [!] 触发长文本截断: Prompt 从 {p_len} 缩减至 {keep_p_len} Tokens")
            
        prompt_len = p_ids.input_ids.shape[1]
        
        # 2. 🚀 Tensor 级物理拼接
        full_ids = torch.cat([p_ids.input_ids, a_ids.input_ids], dim=-1)
        full_mask = torch.cat([p_ids.attention_mask, a_ids.attention_mask], dim=-1)
        seq_len = full_ids.shape[1] # 👈 用于定位 SLT
        
        # 3. 推理
        need_attn = (method == "icr_probe")
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids, 
                attention_mask=full_mask,
                output_hidden_states=True, 
                output_attentions=need_attn
            )
            logits = outputs.logits

        # --- A. 概率补票 (Shift Logits 对齐) ---
        shift_logits = logits[0, :-1, :]
        shift_labels = full_ids[0, 1:]
        log_probs_full = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_logprobs = torch.gather(log_probs_full, index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
        
        # 截取 Answer 段：从索引 prompt_len - 1 开始
        ans_lps = target_logprobs[prompt_len-1:].cpu().float().numpy().astype(np.float16)

        # --- B. 特征提取 ---
        hs_res = {}
        tw_res = {}
        sep_res = {} # 👈 存储 SEP 所需的点特征
        icr_feat = None
        
        # 🎯 ICR Score 计算
        if method == "icr_probe":
            n_hs, n_att = self._pack_for_icr(outputs.hidden_states, outputs.attentions, prompt_len)
            c_pos = {"user_prompt_start": 0, "user_prompt_end": prompt_len - 1, "response_start": prompt_len}
            
            engine = ICRScore(n_hs, n_att, core_positions=c_pos, icr_device=self.device)
            scores, _ = engine.compute_icr(top_k=20, pooling="mean", use_induction_head=False, 
                                          attention_uniform=False, hidden_uniform=False, top_p=None)
            
            means = np.zeros(len(scores), dtype=np.float32)
            for l, v in enumerate(scores):
                means[l] = float(np.mean(v)) if len(v) > 0 else 0.0
            icr_feat = means.astype(np.float16)

        # 隐藏层提取
        target_layers = list(range(self.total_layers)) if extract_all_layers else [self.total_layers - 1]
        for l_idx in target_layers:
            hf_l_idx = l_idx + 1
            all_layer_hs = outputs.hidden_states[hf_l_idx][0] # [seq_len, dim]
            
            # 🎯 普通 Answer 平均特征
            ans_hs = all_layer_hs[prompt_len:, :]
            hs_res[l_idx] = ans_hs.mean(dim=0).cpu().float().numpy().astype(np.float16)
            
            if method == "icr_probe":
                tw_res[l_idx] = ans_hs.cpu().float().numpy().astype(np.float16)

            # 🎯 SEP 关键点提取
            if method == "sep":
                # TBG: Token Before Generation (Prompt 最后一个词)
                tbg_feat = all_layer_hs[prompt_len - 1, :].cpu().float().numpy().astype(np.float16)
                # SLT: Selected Last Token (Answer 最后一个词)
                slt_feat = all_layer_hs[seq_len - 1, :].cpu().float().numpy().astype(np.float16)
                sep_res[f"tbg_layer_{l_idx}"] = tbg_feat
                sep_res[f"slt_layer_{l_idx}"] = slt_feat

        return {
            "logprobs": ans_lps,
            "hidden_states": hs_res,
            "token_wise": tw_res if method == "icr_probe" else None,
            "icr_feature": icr_feat,
            "sep_features": sep_res if method == "sep" else None
        }

def process_dataset(input_jsonl, output_h5, model_name, method, model_kwargs=None, max_samples=None):
    print(f"\n{'='*70}\n🚀 启动 QA 拼接特征提取器 ({method.upper()})\n{'='*70}")
    # 🎯 显式传入 method
    extractor = QAHiddenStateExtractor(model_name=model_name, model_kwargs=model_kwargs, method=method)

    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples: break

    print(f"[*] 准备处理 {len(samples)} 个样本 (断点续传检查中)...")

    with h5py.File(output_h5, 'a') as f_h5:
        for sample in tqdm(samples, desc=f"Inference: {method}"):
            sid = str(sample.get("sample_id"))
            p, a = sample.get("prompt", ""), sample.get("model_output_text", "")
            if not p or not a: continue

            g_name = f"{sid}_{method}"
            # 🛡️ 断点续传：检查数据是否存在
            if g_name in f_h5:
                complete = True
                if method == "icr_probe" and "icr_feature" not in f_h5[g_name]: complete = False
                elif method == "ccs" and "positive" not in f_h5[g_name]: complete = False
                elif method == "sep" and "sep_points" not in f_h5[g_name]: complete = False
                if complete: continue

            try:
                res = extractor.extract_features(p, a, method)
                if g_name in f_h5: del f_h5[g_name]
                grp = f_h5.create_group(g_name)
                grp.create_dataset("logprobs", data=res["logprobs"])
                
                if method == "ccs":
                    # CCS 特殊处理：正负双向
                    res_p = extractor.extract_features(p, a, "ccs")
                    res_n = extractor.extract_features("It is not true that:", f"{p}\n{a}", "ccs")
                    # 此处根据您之前的 CCS 逻辑进行存储... (保持原子逻辑)
                    for mode, r in [("positive", res_p), ("negative", res_n)]:
                        sub = grp.create_group(mode)
                        sub.create_dataset("logprobs", data=r["logprobs"])
                        for l, d in r["hidden_states"].items(): sub.create_dataset(f"layer_{l}", data=d)

                elif method == "icr_probe":
                    grp.create_dataset("icr_feature", data=res["icr_feature"])
                    for sub_n, key in [("averaged", "hidden_states"), ("token_wise", "token_wise")]:
                        sub = grp.create_group(sub_n)
                        for l, d in res[key].items(): sub.create_dataset(f"layer_{l}", data=d)

                elif method == "sep":
                    sub = grp.create_group("sep_points")
                    for k, v in res["sep_features"].items():
                        sub.create_dataset(k, data=v)

                else:
                    # PRISM 或 Base
                    for l, d in res["hidden_states"].items():
                        grp.create_dataset(f"layer_{l}", data=d)

            except Exception as e:
                print(f"\n[!] 样本 {sid} 提取失败: {e}")
                if g_name in f_h5: del f_h5[g_name]
                continue
            finally:
                torch.cuda.empty_cache()

    print(f"\n[✅] 特征提取完毕！产物已存入: {output_h5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_h5", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    process_dataset(args.input_jsonl, args.output_h5, args.model_name, args.method, 
                    {"trust_remote_code": args.trust_remote_code, "attn_implementation": args.attn_implementation})