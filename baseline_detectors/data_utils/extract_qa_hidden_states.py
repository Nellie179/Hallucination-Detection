import os
import json
import torch
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

from data_utils.icr_score import ICRScore


class QAHiddenStateExtractor:
    def __init__(
            self,
            model_name: str,
            device: str = None,
            dtype: torch.dtype = torch.bfloat16,
            model_kwargs: dict = None,
            method: str = None,
            model=None,
            tokenizer=None,
            pooling: str = "mean"
    ):
        self.method = method
        self.model_name = model_name
        self.model_kwargs = (model_kwargs or {}).copy()
        self.pooling = pooling

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            print(
                f"[*] QAHiddenStateExtractor ({method.upper()}) successfully received injected model instance (Device: {self.device})")

            if self.method == "icr_probe" and getattr(self.model.config, "_attn_implementation", "") != "eager":
                print("[!] ⚠️ Architecture Warning: icr_probe heavily relies on Attention matrices!")
                print(
                    "[!] The currently injected global model might be using sdpa or flash_attention, which will prevent Attention extraction.")
                print(
                    "[!] If an error occurs later, please change attn_implementation to 'eager' in EVAL_CONFIG of runner.py!")
        else:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            if self.method == "icr_probe":
                if self.model_kwargs.get("attn_implementation") != "eager":
                    print(f"[*] Note: {self.method} requires Attention matrices, switching implementation to eager...")
                    self.model_kwargs["attn_implementation"] = "eager"

            print(f"[*] Loading model {model_name} to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **self.model_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device,
                **self.model_kwargs
            ).eval()
            print(f"[+] Independent model loaded successfully.")

        if hasattr(self.model.config, "num_hidden_layers"):
            self.total_layers = self.model.config.num_hidden_layers
        else:
            self.total_layers = self.model.config.text_config.num_hidden_layers

    def _pack_for_icr(self, hidden_states, attentions, prompt_len):
        if attentions is None or len(attentions) == 0:
            raise ValueError("Failed to obtain Attention matrices. Please ensure attn_implementation is 'eager'.")

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
            if pos >= S: break

            token_hs = [hs_tuple[ell][0, pos:pos + 1, :].unsqueeze(0) for ell in range(actual_hs_layers)]
            nested_hs.append(token_hs)

        nested_att = []
        if actual_att_layers > 0:
            att_input = [att_tuple[ell][0].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(att_input)

        for t in range(output_size):
            pos = prompt_len + t
            if pos >= S: break

            token_att = [att_tuple[ell][0, :, pos:pos + 1, :].unsqueeze(0) for ell in range(actual_att_layers)]
            nested_att.append(token_att)

        return nested_hs, nested_att

    def extract_features(
            self,
            prompt: str,
            answer: str,
            method: str,
            extract_all_layers: bool = True
    ) -> Dict[str, Any]:

        MAX_SEQ_LEN = 1536 if method == "icr_probe" else 2048

        if method == "prism":
            statement = f"{prompt}\n{answer}".strip()
            prism_prompt = f"Does the statement '{statement}' accurately reflect the truth?"

            original_side = self.tokenizer.truncation_side
            self.tokenizer.truncation_side = 'left'
            inputs = self.tokenizer(prism_prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                                    max_length=MAX_SEQ_LEN).to(self.device)
            self.tokenizer.truncation_side = original_side

            full_ids = inputs.input_ids
            full_mask = inputs.attention_mask
            prompt_len = full_ids.shape[1]
            seq_len = full_ids.shape[1]

        elif method == "self_evaluator":
            target_prompt = f"Question: {prompt}\nProposed Answer: {answer}\nIs the proposed answer True or False?\nAnswer:"

            original_side = self.tokenizer.truncation_side
            self.tokenizer.truncation_side = 'left'
            inputs = self.tokenizer(target_prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                                    max_length=MAX_SEQ_LEN).to(self.device)
            self.tokenizer.truncation_side = original_side

            full_ids = inputs.input_ids
            full_mask = inputs.attention_mask
            prompt_len = full_ids.shape[1]
            seq_len = full_ids.shape[1]

        else:
            a_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(self.device)
            ans_len = a_ids.input_ids.shape[1]

            if ans_len >= MAX_SEQ_LEN - 32:
                a_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True,
                                       max_length=MAX_SEQ_LEN - 32).to(self.device)
                ans_len = a_ids.input_ids.shape[1]

            remaining_len = MAX_SEQ_LEN - ans_len

            original_side = self.tokenizer.truncation_side
            self.tokenizer.truncation_side = 'left'
            p_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                                   max_length=remaining_len).to(self.device)
            self.tokenizer.truncation_side = original_side

            full_ids = torch.cat([p_ids.input_ids, a_ids.input_ids], dim=-1)
            full_mask = torch.cat([p_ids.attention_mask, a_ids.attention_mask], dim=-1)

            prompt_len = p_ids.input_ids.shape[1]
            seq_len = full_ids.shape[1]

        need_attn = (method == "icr_probe")
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
                output_attentions=need_attn,
                use_cache=False
            )
            logits = outputs.logits

        ans_lps = None
        if method not in ["prism", "self_evaluator"]:
            shift_logits = logits[0, :-1, :]
            shift_labels = full_ids[0, 1:]
            log_probs_full = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            target_logprobs = torch.gather(log_probs_full, index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
            start_idx = max(0, prompt_len - 1)
            ans_lps = target_logprobs[start_idx:].cpu().float().numpy().astype(np.float32)

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
                        math.exp(logit_true - max_logit) + math.exp(logit_false - max_logit) + 1e-9)

            ans_lps = np.array([math.log(max(prob_true, 1e-9))]).astype(np.float32)

        hs_res = {}
        tw_res = {}
        sep_res = {}
        icr_feat = None

        if method == "icr_probe":
            n_hs, n_att = self._pack_for_icr(outputs.hidden_states, outputs.attentions, prompt_len)
            c_pos = {"user_prompt_start": 0, "user_prompt_end": prompt_len - 1, "response_start": prompt_len}

            engine = ICRScore(n_hs, n_att, core_positions=c_pos, icr_device=self.device)
            scores, _ = engine.compute_icr(top_k=20, pooling="mean", use_induction_head=False,
                                           attention_uniform=False, hidden_uniform=False, top_p=None)

            means = np.zeros(len(scores), dtype=np.float32)
            for l, v in enumerate(scores):
                means[l] = float(np.mean(v)) if len(v) > 0 else 0.0
            icr_feat = means.astype(np.float32)

        target_layers = list(range(self.total_layers)) if extract_all_layers else [self.total_layers - 1]
        for l_idx in target_layers:
            hf_l_idx = l_idx + 1
            all_layer_hs = outputs.hidden_states[hf_l_idx][0]

            if method == "prism":
                hs_res[l_idx] = all_layer_hs[-1, :].cpu().float().numpy().astype(np.float32)
            else:
                ans_hs = all_layer_hs[prompt_len:, :]
                if self.pooling == "last":
                    target_hs = ans_hs[-1, :]
                else:
                    target_hs = ans_hs.mean(dim=0)

                hs_res[l_idx] = target_hs.cpu().float().numpy().astype(np.float32)

            if method == "icr_probe":
                tw_res[l_idx] = ans_hs.cpu().float().numpy().astype(np.float32)

            if method == "sep":
                tbg_feat = all_layer_hs[prompt_len - 1, :].cpu().float().numpy().astype(np.float32)
                slt_feat = all_layer_hs[seq_len - 1, :].cpu().float().numpy().astype(np.float32)
                sep_res[f"tbg_layer_{l_idx}"] = tbg_feat
                sep_res[f"slt_layer_{l_idx}"] = slt_feat

        return {
            "logprobs": ans_lps,
            "hidden_states": hs_res,
            "token_wise": tw_res if method == "icr_probe" else None,
            "icr_feature": icr_feat,
            "sep_features": sep_res if method == "sep" else None
        }


def process_dataset(input_jsonl, output_h5, model_name, method, model_kwargs=None, max_samples=None, model=None,
                    tokenizer=None, pooling="mean"):
    print(f"\n{'=' * 70}\n🚀 Starting QA Concatenation Feature Extractor ({method.upper()})\n{'=' * 70}")

    extractor = QAHiddenStateExtractor(
        model_name=model_name,
        model_kwargs=model_kwargs,
        method=method,
        model=model,
        tokenizer=tokenizer,
        pooling=pooling
    )

    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples: break

    print(f"[*] Preparing to process {len(samples)} samples (checking for resumption)...")

    with h5py.File(output_h5, 'a') as f_h5:
        for sample in tqdm(samples, desc=f"Inference: {method}"):
            sid = str(sample.get("sample_id"))
            p, a = sample.get("prompt", ""), sample.get("model_output_text", "")
            if not p or not a: continue

            g_name = f"{sid}_{method}"
            if g_name in f_h5:
                complete = True
                if method == "icr_probe" and "icr_feature" not in f_h5[g_name]:
                    complete = False
                elif method == "ccs" and "positive" not in f_h5[g_name]:
                    complete = False
                elif method == "sep" and "sep_points" not in f_h5[g_name]:
                    complete = False
                if complete: continue

            try:
                res = extractor.extract_features(p, a, method)
                if g_name in f_h5: del f_h5[g_name]
                grp = f_h5.create_group(g_name)

                if method != "prism":
                    grp.create_dataset("logprobs", data=res["logprobs"])

                if method == "ccs":
                    res_p = extractor.extract_features(p, a, "ccs")
                    res_n = extractor.extract_features("It is not true that:", f"{p}\n{a}", "ccs")
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
                    for l, d in res["hidden_states"].items():
                        grp.create_dataset(f"layer_{l}", data=d)

            except Exception as e:
                error_msg = str(e).lower()
                print(f"\n[!] Failed to extract sample {sid}: {e}")

                if g_name in f_h5:
                    del f_h5[g_name]
                    f_h5.flush()

                error_log_path = output_h5 + ".failed_ids.txt"
                with open(error_log_path, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"{sid}\t{e}\n")

                if "out of memory" in error_msg or "oom" in error_msg:
                    print(
                        f"    [!] 🚨 OOM Defense Triggered: Logged to dead letter queue, forcefully clearing VRAM, rescuing pipeline...")
                    torch.cuda.empty_cache()

                continue
            finally:
                torch.cuda.empty_cache()

    print(f"\n[✅] Feature extraction complete! Outputs saved to: {output_h5}")


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