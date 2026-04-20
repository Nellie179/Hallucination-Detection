# icr_patch_missing_samples.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from Benchmark.baseline_detectors.data_utils.icr_score import ICRScore   # 保证你的 icr_score.py 在 PYTHONPATH 下

# ====== 基础配置（按你现在的环境来） ======
ACCESS_TOKEN = "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"
MODEL_PATH   = "meta-llama/Meta-Llama-3.1-8B"
MODEL_TAG    = "llama3.1-8B"

# 原来已经算好的 ICR 特征目录（你自己改成之前的那个，比如 icr_feats_llama3.1-8B_1）
OLD_OUT_DIR  = "./icr_feats_llama3.1-8B"

# 补完之后的新目录（不会覆盖老的）
NEW_OUT_DIR  = "./icr_feats_llama3.1-8B_patched"

# 新生成的 answers 放在这里（就是你 regenerate_nq / regenerate_triviaqa 用的这个）
ANS_BASE_NEW = "/home/zfang1/Data/Lxy/Causal_generalize/save_for_eval_new"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K        = 20
USE_INDUCTION_HEAD = False
TRUNCATE_MAXLEN    = 512

download_config = DownloadConfig(use_auth_token=os.getenv("HF_TOKEN"))

# 你刚才查出来的缺失样本：
MISSING_IDX = {
    'math': [27, 71, 138, 343, 380, 381, 419, 464, 467, 532, 639, 686, 703, 717, 722, 774, 823, 898, 1010, 1015],
    'mgsm': [43,44],
    'SVAMP': [924]
}
# ==========================================


def _load_dataset(dt):
    if dt == "tqa":
        ds = load_dataset("truthful_qa", "generation")["validation"]
    elif dt == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch["question_id"][0] in id_mem:
                return {k: [] for k in batch.keys()}
            id_mem.add(batch["question_id"][0])
            return batch

        ds = ds.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif dt == "sciq":
        ds = load_dataset("allenai/sciq", split="validation", download_config=download_config)
    elif dt == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open",
                          split="validation",
                          download_config=download_config)
    else:
        raise ValueError(f"Unknown dataset: {dt}")
    return ds


def _get_question(ds, i):
    return ds[i]["question"]


def _answers_path_new(dt, model_tag, i):
    return os.path.join(
        ANS_BASE_NEW,
        f"{dt}_hal_det/answers/most_likely_hal_det_{model_tag}_{dt}_answers_index_{i}.npy",
    )


def _load_answers(path, max_answers=None):
    if not os.path.exists(path):
        print("啥都没有")
        return []
    arr = np.load(path, allow_pickle=True)
    answers = list(arr)
    if max_answers is not None and len(answers) > max_answers:
        answers = answers[:max_answers]
    answers = [a.decode("utf-8", errors="replace") if isinstance(a, bytes) else str(a) for a in answers]
    return answers



@torch.no_grad()
def encode_qa(tokenizer, question: str, answer: str):
    sep = "\n\n"
    user_only = f"User: {question}{sep}"
    qa_text   = f"{user_only}Assistant: {answer}"

    enc_user = tokenizer(
        user_only,
        return_tensors="pt",
        truncation=True,
        max_length=TRUNCATE_MAXLEN,
    )
    enc_qa   = tokenizer(
        qa_text,
        return_tensors="pt",
        truncation=True,
        max_length=TRUNCATE_MAXLEN,
    )

    input_ids = enc_qa["input_ids"]
    response_start = enc_user["input_ids"].shape[1]
    positions = dict(
        user_prompt_start=0,
        user_prompt_end=response_start - 1,
        response_start=response_start,
    )
    return input_ids, positions


@torch.no_grad()
def forward_full(model, input_ids: torch.Tensor, device: str):
    out = model(
        input_ids=input_ids.to(device),
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    hs_tuple  = tuple(h.detach().cpu() for h in out.hidden_states)
    att_tuple = tuple(a.detach().cpu() for a in out.attentions)
    return hs_tuple, att_tuple


def pack_for_icr_from_full_forward(hidden_states_tuple, attentions_tuple, response_start, batch_idx=0):
    L_plus_1 = len(hidden_states_tuple)
    L = L_plus_1 - 1
    B, S, D = hidden_states_tuple[0].shape
    input_len   = response_start
    output_size = max(0, S - input_len)

    nested_hidden_states = []
    hs_input_per_layer = []
    for ell in range(L_plus_1):
        h = hidden_states_tuple[ell][batch_idx, :input_len, :]
        hs_input_per_layer.append(h.unsqueeze(0))
    nested_hidden_states.append(hs_input_per_layer)

    nested_attentions = []
    attn_input_per_layer = []
    for ell in range(L):
        A = attentions_tuple[ell][batch_idx]
        attn_input_per_layer.append(A.unsqueeze(0))
    nested_attentions.append(attn_input_per_layer)

    for t in range(output_size):
        pos = input_len + t
        hs_t_per_layer = []
        attn_t_per_layer = []

        for ell in range(L_plus_1):
            h = hidden_states_tuple[ell][batch_idx, pos:pos+1, :]
            hs_t_per_layer.append(h.unsqueeze(0))
        nested_hidden_states.append(hs_t_per_layer)

        for ell in range(L):
            A = attentions_tuple[ell][batch_idx]
            row = A[:, pos:pos+1, :]
            attn_t_per_layer.append(row.unsqueeze(0))
        nested_attentions.append(attn_t_per_layer)

    return nested_hidden_states, nested_attentions


def layer_means_from_icr_scores(icr_scores_item):
    L = len(icr_scores_item)
    means = np.zeros(L, dtype=np.float32)
    for l in range(L):
        vals = icr_scores_item[l]
        means[l] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return means


def main():
    os.makedirs(NEW_OUT_DIR, exist_ok=True)

    # 统一加载 tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        token=ACCESS_TOKEN,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        token=ACCESS_TOKEN,
    ).eval()
    torch.set_grad_enabled(False)

    for dt, idx_list in MISSING_IDX.items():
        print(f"=== Patching dataset: {dt} ===")
        # 1) 读原来的特征
        old_npz_path = os.path.join(OLD_OUT_DIR, f"{dt}_icr_features.npz")
        data = np.load(old_npz_path, allow_pickle=True)
        X_old   = data["X"]
        meta_old = data["meta"]

        print(f"  Old X shape: {X_old.shape}, meta shape: {meta_old.shape}")

        # 2) 准备数据集
        ds = _load_dataset(dt)

        # 3) 只对缺失 q_idx 做 ICR
        new_feats = []
        new_metas = []

        for i in tqdm(idx_list, desc=f"{dt} missing"):
            q = _get_question(ds, i)
            ans_path = _answers_path_new(dt, MODEL_TAG, i)
            answers = _load_answers(ans_path)
            if not answers:
                print(f"[warn] {dt} q_idx={i} has no answer in new folder, skip.")
                continue

            # 你现在每个样本只有一个答案，这里就拿 answers[0]
            anw = answers[0]

            try:
                input_ids, positions = encode_qa(tokenizer, q, anw)
                hs_tuple, att_tuple = forward_full(model, input_ids, device=DEVICE)
                nested_hs, nested_attn = pack_for_icr_from_full_forward(
                    hidden_states_tuple=hs_tuple,
                    attentions_tuple=att_tuple,
                    response_start=positions["response_start"],
                    batch_idx=0,
                )
                icr = ICRScore(
                    hidden_states=nested_hs,
                    attentions=nested_attn,
                    core_positions=positions,
                    icr_device=("cuda" if DEVICE == "cuda" else "cpu"),
                    skew_threshold=3,
                    entropy_threshold=3,
                )
                icr_scores_item, _ = icr.compute_icr(
                    top_k=TOP_K,
                    top_p=None,
                    pooling="mean",
                    attention_uniform=False,
                    hidden_uniform=False,
                    use_induction_head=USE_INDUCTION_HEAD,
                )
                feat = layer_means_from_icr_scores(icr_scores_item)
                new_feats.append(feat)
                new_metas.append({"q_idx": i, "a_idx": 0})
            finally:
                del input_ids
                if "hs_tuple" in locals():
                    del hs_tuple
                if "att_tuple" in locals():
                    del att_tuple
                if "nested_hs" in locals():
                    del nested_hs
                if "nested_attn" in locals():
                    del nested_attn
                if "icr" in locals():
                    del icr

        if new_feats:
            X_new_part = np.stack(new_feats, axis=0)
            meta_new_part = np.array(new_metas, dtype=object)

            X_merged    = np.concatenate([X_old, X_new_part], axis=0)
            meta_merged = np.concatenate([meta_old, meta_new_part], axis=0)
        else:
            X_merged, meta_merged = X_old, meta_old

        # 4) 存到新的目录
        out_path = os.path.join(NEW_OUT_DIR, f"{dt}_icr_features.npz")
        np.savez_compressed(out_path, X=X_merged, meta=meta_merged)
        print(f"  Saved patched features to: {out_path}")
        print(f"  New X shape: {X_merged.shape}, meta shape: {meta_merged.shape}")

    print("All patching done.")


if __name__ == "__main__":
    main()
