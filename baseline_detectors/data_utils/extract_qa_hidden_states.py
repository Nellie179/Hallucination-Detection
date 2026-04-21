"""
Extract hidden states, LOGPROBS and ICR Scores from Q+A (Universal Version)
权威拼接版：基于 Tensor 物理拼接，确保 Token 绝对对齐并防止索引越界。
已新增 SEP (Semantic Entropy Probes) 和 PRISM (Prompt-guided) 特征提取逻辑。
支持全局依赖注入 (Dependency Injection)，极致优化显存。
"""

import os
import json
import math
import logging
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils.icr_score import ICRScore

logger = logging.getLogger(__name__)


class QAHiddenStateExtractor:
    """支持全量张量提取与概率截获的终极提取器"""

    def __init__(
        self,
        model_name: str,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        model_kwargs: dict = None,
        method: str = None,
        model=None,
        tokenizer=None,
        pooling: str = "mean",
    ):
        self.method = method
        self.model_name = model_name
        self.model_kwargs = (model_kwargs or {}).copy()
        self.pooling = pooling

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device

            if self.method == "icr_probe" and getattr(self.model.config, "_attn_implementation", "") != "eager":
                logger.warning(
                    "icr_probe requires the attention matrix, but the injected model uses "
                    "sdpa / flash_attention. Set attn_implementation='eager' in EVAL_CONFIG if this fails."
                )
        else:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            if self.method == "icr_probe" and self.model_kwargs.get("attn_implementation") != "eager":
                self.model_kwargs["attn_implementation"] = "eager"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **self.model_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device,
                **self.model_kwargs,
            ).eval()

        if hasattr(self.model.config, "num_hidden_layers"):
            self.total_layers = self.model.config.num_hidden_layers
        else:
            self.total_layers = self.model.config.text_config.num_hidden_layers

    def _pack_for_icr(self, hidden_states, attentions, prompt_len):
        """将 HF 输出转换为 ICRScore 结构，使用动态长度感应，防止层数不匹配"""
        if attentions is None or len(attentions) == 0:
            raise ValueError("未能获取到 Attention 矩阵。请确认 attn_implementation 为 'eager'。")

        hs_tuple = tuple(h.detach() for h in hidden_states)
        att_tuple = tuple(a.detach() for a in attentions)

        actual_hs_layers = len(hs_tuple)
        actual_att_layers = len(att_tuple)

        B, S, D = hs_tuple[0].shape
        output_size = max(0, S - prompt_len)

        nested_hs = []
        if actual_hs_layers > 0:
            hs_input = [hs_tuple[ell][0, :prompt_len, :].unsqueeze(0) for ell in range(actual_hs_layers)]
            nested_hs.append(hs_input)

        for t in range(output_size):
            pos = prompt_len + t
            if pos >= S:
                break
            token_hs = [hs_tuple[ell][0, pos:pos+1, :].unsqueeze(0) for ell in range(actual_hs_layers)]
            nested_hs.append(token_hs)

        nested_att = []
        if actual_att_layers > 0:
            att_input = [att_tuple[ell][0].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(att_input)

        for t in range(output_size):
            pos = prompt_len + t
            if pos >= S:
                break
            token_att = [att_tuple[ell][0, :, pos:pos+1, :].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(token_att)

        return nested_hs, nested_att

    def extract_features_batch(
        self,
        prompts: List[str],
        answers: List[str],
        method: str,
    ) -> List[Dict[str, Any]]:
        """Batched forward pass for the default Q+A concat branch of extract_features.

        Only supports methods that go through the generic P+A concat path — namely
        everything except 'prism', 'self_evaluator', 'icr_probe', 'sep', which have
        special tokenization or feature-extraction logic handled per-sample upstream.

        Uses RIGHT-padding under causal attention: real tokens never attend to pads,
        so per-row hidden states are identical to the single-sample forward pass up
        to GEMM non-associativity (≤1e-4 drift in bf16).
        """
        assert method not in ("prism", "self_evaluator", "icr_probe", "sep"), (
            f"extract_features_batch does not support method={method!r}; use extract_features"
        )

        B = len(prompts)
        full_ids_list: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        real_lens: List[int] = []
        for p, a in zip(prompts, answers):
            p_ids = self.tokenizer(p, return_tensors="pt", add_special_tokens=True).input_ids[0]
            a_ids = self.tokenizer(a, return_tensors="pt", add_special_tokens=False).input_ids[0]
            full_ids = torch.cat([p_ids, a_ids], dim=0)
            full_ids_list.append(full_ids)
            prompt_lens.append(int(p_ids.shape[0]))
            real_lens.append(int(full_ids.shape[0]))

        max_len = max(real_lens)
        pad_id = self.tokenizer.pad_token_id
        padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        padded_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, t in enumerate(full_ids_list):
            padded_ids[i, : real_lens[i]] = t
            padded_mask[i, : real_lens[i]] = 1

        padded_ids = padded_ids.to(self.device)
        padded_mask = padded_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=padded_ids,
                attention_mask=padded_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits

        target_layers = list(range(self.total_layers))
        results: List[Dict[str, Any]] = []
        for b in range(B):
            real_len = real_lens[b]
            plen = prompt_lens[b]

            shift_logits = logits[b, : real_len - 1, :]
            shift_labels = padded_ids[b, 1:real_len]
            log_probs_full = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            target_logprobs = torch.gather(
                log_probs_full, index=shift_labels.unsqueeze(-1), dim=-1
            ).squeeze(-1)
            start_idx = max(0, plen - 1)
            ans_lps = target_logprobs[start_idx:].cpu().float().numpy().astype(np.float16)

            hs_res: Dict[int, np.ndarray] = {}
            for l_idx in target_layers:
                hf_l_idx = l_idx + 1
                all_layer_hs = outputs.hidden_states[hf_l_idx][b]
                ans_hs = all_layer_hs[plen:real_len, :]
                if self.pooling == "last":
                    target_hs = ans_hs[-1, :]
                else:
                    target_hs = ans_hs.mean(dim=0)
                hs_res[l_idx] = target_hs.cpu().float().numpy().astype(np.float16)

            results.append({
                "logprobs": ans_lps,
                "hidden_states": hs_res,
                "token_wise": None,
                "icr_feature": None,
                "sep_features": None,
            })

        del outputs, logits
        return results

    def extract_features(
        self,
        prompt: str,
        answer: str,
        method: str,
        extract_all_layers: bool = True,
    ) -> Dict[str, Any]:

        if method == "prism":
            statement = f"{prompt}\n{answer}".strip()
            prism_prompt = f"Does the statement '{statement}' accurately reflect the truth?"
            full_ids = self.tokenizer(prism_prompt, return_tensors="pt", add_special_tokens=True).to(self.device).input_ids
            full_mask = self.tokenizer(prism_prompt, return_tensors="pt", add_special_tokens=True).to(self.device).attention_mask
            prompt_len = full_ids.shape[1]
            seq_len = full_ids.shape[1]

        elif method == "self_evaluator":
            target_prompt = f"Question: {prompt}\nProposed Answer: {answer}\nIs the proposed answer True or False?\nAnswer:"
            full_ids = self.tokenizer(target_prompt, return_tensors="pt", add_special_tokens=True).to(self.device).input_ids
            full_mask = self.tokenizer(target_prompt, return_tensors="pt", add_special_tokens=True).to(self.device).attention_mask
            prompt_len = full_ids.shape[1]
            seq_len = full_ids.shape[1]

        else:
            p_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            a_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(self.device)
            prompt_len = p_ids.input_ids.shape[1]
            full_ids = torch.cat([p_ids.input_ids, a_ids.input_ids], dim=-1)
            full_mask = torch.cat([p_ids.attention_mask, a_ids.attention_mask], dim=-1)
            seq_len = full_ids.shape[1]

        need_attn = (method == "icr_probe")
        with torch.inference_mode():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
                output_attentions=need_attn,
            )
            logits = outputs.logits

        ans_lps = None
        if method not in ["prism", "self_evaluator"]:
            shift_logits = logits[0, :-1, :]
            shift_labels = full_ids[0, 1:]
            log_probs_full = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            target_logprobs = torch.gather(log_probs_full, index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
            start_idx = max(0, prompt_len - 1)
            ans_lps = target_logprobs[start_idx:].cpu().float().numpy().astype(np.float16)

        if method == "self_evaluator":
            next_token_logits = logits[0, -1, :]
            true_token_ids = self.tokenizer.encode("True", add_special_tokens=False)
            false_token_ids = self.tokenizer.encode("False", add_special_tokens=False)

            id_true = true_token_ids[-1] if true_token_ids else 0
            id_false = false_token_ids[-1] if false_token_ids else 0

            logit_true = next_token_logits[id_true].item()
            logit_false = next_token_logits[id_false].item()

            max_logit = max(logit_true, logit_false)
            prob_true = math.exp(logit_true - max_logit) / (
                math.exp(logit_true - max_logit) + math.exp(logit_false - max_logit) + 1e-9
            )
            ans_lps = np.array([math.log(max(prob_true, 1e-9))]).astype(np.float16)

        hs_res: Dict[int, np.ndarray] = {}
        tw_res: Dict[int, np.ndarray] = {}
        sep_res: Dict[str, np.ndarray] = {}
        icr_feat = None

        if method == "icr_probe":
            n_hs, n_att = self._pack_for_icr(outputs.hidden_states, outputs.attentions, prompt_len)
            c_pos = {"user_prompt_start": 0, "user_prompt_end": prompt_len - 1, "response_start": prompt_len}

            engine = ICRScore(n_hs, n_att, core_positions=c_pos, icr_device=self.device)
            scores, _ = engine.compute_icr(
                top_k=20, pooling="mean", use_induction_head=False,
                attention_uniform=False, hidden_uniform=False, top_p=None,
            )

            means = np.zeros(len(scores), dtype=np.float32)
            for l, v in enumerate(scores):
                means[l] = float(np.mean(v)) if len(v) > 0 else 0.0
            icr_feat = means.astype(np.float16)

        target_layers = list(range(self.total_layers)) if extract_all_layers else [self.total_layers - 1]
        for l_idx in target_layers:
            hf_l_idx = l_idx + 1
            all_layer_hs = outputs.hidden_states[hf_l_idx][0]

            if method == "prism":
                hs_res[l_idx] = all_layer_hs[-1, :].cpu().float().numpy().astype(np.float16)
            else:
                ans_hs = all_layer_hs[prompt_len:, :]
                if self.pooling == "last":
                    target_hs = ans_hs[-1, :]
                else:
                    target_hs = ans_hs.mean(dim=0)
                hs_res[l_idx] = target_hs.cpu().float().numpy().astype(np.float16)

            if method == "icr_probe":
                tw_res[l_idx] = ans_hs.cpu().float().numpy().astype(np.float16)

            if method == "sep":
                tbg_feat = all_layer_hs[prompt_len - 1, :].cpu().float().numpy().astype(np.float16)
                slt_feat = all_layer_hs[seq_len - 1, :].cpu().float().numpy().astype(np.float16)
                sep_res[f"tbg_layer_{l_idx}"] = tbg_feat
                sep_res[f"slt_layer_{l_idx}"] = slt_feat

        return {
            "logprobs": ans_lps,
            "hidden_states": hs_res,
            "token_wise": tw_res if method == "icr_probe" else None,
            "icr_feature": icr_feat,
            "sep_features": sep_res if method == "sep" else None,
        }


_BATCHABLE_METHODS = {"base_logit_recovery", "saplma", "haloscope", "mind"}


def _write_default_h5_group(f_h5, sid: str, method: str, res: Dict[str, Any]):
    g_name = f"{sid}_{method}"
    if g_name in f_h5:
        del f_h5[g_name]
    grp = f_h5.create_group(g_name)
    grp.create_dataset("logprobs", data=res["logprobs"])
    for l, d in res["hidden_states"].items():
        grp.create_dataset(f"layer_{l}", data=d)


def _log_sample_failure(output_h5: str, sid: str, exc: Exception):
    error_log_path = output_h5 + ".failed_ids.txt"
    with open(error_log_path, 'a', encoding='utf-8') as f_err:
        f_err.write(f"{sid}\t{exc}\n")


def process_dataset(
    input_jsonl,
    output_h5,
    model_name,
    method,
    model_kwargs=None,
    max_samples=None,
    model=None,
    tokenizer=None,
    pooling="mean",
):
    extractor = QAHiddenStateExtractor(
        model_name=model_name,
        model_kwargs=model_kwargs,
        method=method,
        model=model,
        tokenizer=tokenizer,
        pooling=pooling,
    )

    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break

    use_batched = method in _BATCHABLE_METHODS
    batch_size = max(1, int(os.getenv("QA_BATCH", "2")))

    with h5py.File(output_h5, 'a') as f_h5:
        if use_batched and batch_size > 1:
            pending = []
            for sample in samples:
                sid = str(sample.get("sample_id"))
                p, a = sample.get("prompt", ""), sample.get("model_output_text", "")
                if not p or not a:
                    continue
                if f"{sid}_{method}" in f_h5:
                    continue
                pending.append((sid, p, a))

            logger.info(
                f"[{method}] batched QA extraction: {len(pending)} pending, batch_size={batch_size}"
            )

            def _run_single(sid, p, a):
                try:
                    res = extractor.extract_features(p, a, method)
                    _write_default_h5_group(f_h5, sid, method, res)
                except Exception as e2:
                    logger.warning(f"sample {sid} extraction failed (single-sample fallback): {e2}")
                    if f"{sid}_{method}" in f_h5:
                        del f_h5[f"{sid}_{method}"]
                    _log_sample_failure(output_h5, sid, e2)
                    if "out of memory" in str(e2).lower():
                        torch.cuda.empty_cache()

            pbar = tqdm(range(0, len(pending), batch_size), desc=f"Inference: {method} (batch={batch_size})")
            for i in pbar:
                chunk = pending[i : i + batch_size]
                sids = [x[0] for x in chunk]
                prompts = [x[1] for x in chunk]
                answers = [x[2] for x in chunk]

                try:
                    batch_results = extractor.extract_features_batch(prompts, answers, method)
                    for (sid, _, _), res in zip(chunk, batch_results):
                        try:
                            _write_default_h5_group(f_h5, sid, method, res)
                        except Exception as ewrite:
                            logger.warning(f"H5 write failed for {sid}: {ewrite}")
                            _log_sample_failure(output_h5, sid, ewrite)
                    f_h5.flush()
                except Exception as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "oom" in error_msg:
                        logger.warning(
                            f"OOM on batch starting {sids[0]} (size={len(chunk)}) — clearing cache and falling back to single-sample"
                        )
                        torch.cuda.empty_cache()
                        for sid, p, a in chunk:
                            _run_single(sid, p, a)
                        f_h5.flush()
                        torch.cuda.empty_cache()
                    else:
                        logger.warning(f"batch extraction failed for {sids}: {e}")
                        for sid, p, a in chunk:
                            _run_single(sid, p, a)
                        f_h5.flush()
            return

        for sample in tqdm(samples, desc=f"Inference: {method}"):
            sid = str(sample.get("sample_id"))
            p, a = sample.get("prompt", ""), sample.get("model_output_text", "")
            if not p or not a:
                continue

            g_name = f"{sid}_{method}"
            if g_name in f_h5:
                complete = True
                if method == "icr_probe" and "icr_feature" not in f_h5[g_name]:
                    complete = False
                elif method == "ccs" and "positive" not in f_h5[g_name]:
                    complete = False
                elif method == "sep" and "sep_points" not in f_h5[g_name]:
                    complete = False
                if complete:
                    continue

            try:
                res = extractor.extract_features(p, a, method)
                if g_name in f_h5:
                    del f_h5[g_name]
                grp = f_h5.create_group(g_name)

                if method != "prism":
                    grp.create_dataset("logprobs", data=res["logprobs"])

                if method == "ccs":
                    res_p = extractor.extract_features(p, a, "ccs")
                    res_n = extractor.extract_features("It is not true that:", f"{p}\n{a}", "ccs")
                    for mode, r in [("positive", res_p), ("negative", res_n)]:
                        sub = grp.create_group(mode)
                        sub.create_dataset("logprobs", data=r["logprobs"])
                        for l, d in r["hidden_states"].items():
                            sub.create_dataset(f"layer_{l}", data=d)

                elif method == "icr_probe":
                    grp.create_dataset("icr_feature", data=res["icr_feature"])
                    for sub_n, key in [("averaged", "hidden_states"), ("token_wise", "token_wise")]:
                        sub = grp.create_group(sub_n)
                        for l, d in res[key].items():
                            sub.create_dataset(f"layer_{l}", data=d)

                elif method == "sep":
                    sub = grp.create_group("sep_points")
                    for k, v in res["sep_features"].items():
                        sub.create_dataset(k, data=v)

                else:
                    for l, d in res["hidden_states"].items():
                        grp.create_dataset(f"layer_{l}", data=d)

            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"sample {sid} extraction failed: {e}")

                # 物理铲除残缺的 H5 Group，杜绝脏数据
                if g_name in f_h5:
                    del f_h5[g_name]
                    f_h5.flush()

                # 死信队列 (DLQ)：失败的 ID 写入日志
                _log_sample_failure(output_h5, sid, e)

                # OOM 显存急救
                if "out of memory" in error_msg or "oom" in error_msg:
                    logger.warning(f"OOM on sample {sid} — clearing CUDA cache and continuing.")
                    torch.cuda.empty_cache()

                continue
            finally:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_h5", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_dataset(
        args.input_jsonl,
        args.output_h5,
        args.model_name,
        args.method,
        {"trust_remote_code": args.trust_remote_code, "attn_implementation": args.attn_implementation},
    )
