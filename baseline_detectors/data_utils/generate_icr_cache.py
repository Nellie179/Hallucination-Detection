# icr_extract_features_per_dataset.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============== 固定环境/模型/路径配置 ===============
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 如果你有单独的 HF 缓存目录，也可以加上：
# os.environ["HF_HOME"] = "/root/autodl-tmp/cache/"

access_token = os.getenv("HF_TOKEN")
download_config = DownloadConfig(use_auth_token=access_token)
HF_NAMES = {
    "Llama3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "qwen2.5-14B": "Qwen/Qwen2.5-14B",
}


ACCESS_TOKEN = "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"  # 完全本地可去掉
MODEL_PATH   = "Qwen/Qwen2.5-14B"           # Llama-3.1-8B Base
MODEL_TAG    = "qwen2.5-14B"                            # 只用于 answers 路径/命名
OUT_DIR      = f"./icr_feats_{MODEL_TAG}_QA"               # 输出目录（每数据集一个文件）
ANS_BASE     = "/home/zfang1/Data/Lxy/Causal_generalize/save_for_eval"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS     = ["tqa", "sciq", "triviaqa", "nq_open"]   # 要处理的数据集
TOP_K        = 20          # ICR: 只取 top-k 注意力位置
USE_INDUCTION_HEAD = False # 是否用 induction 头筛选
TRUNCATE_MAXLEN = 512      # 限制长度（注意力是 O(S^2)，太长会非常慢）
MAX_ANSWERS_PER_QUESTION = 1  # 每个问题最多取多少个答案；你现在只有一个，就设 1
# =====================================================

from Benchmark.baseline_detectors.data_utils.icr_score import ICRScore


@torch.no_grad()
def encode_qa(tokenizer, question: str, answer: str):
    """
    Base 模型：手动拼接输入
    User: {Q}\n\nAssistant: {A}
    返回 input_ids 与 positions['response_start']。
    """
    sep = "\n\n"
    user_only = f"User: {question}{sep}"
    qa_text   = f"{user_only}Assistant: {answer}"

    enc_user = tokenizer(
        user_only,
        return_tensors="pt",
        truncation=(TRUNCATE_MAXLEN is not None),
        max_length=TRUNCATE_MAXLEN,
    )
    enc_qa   = tokenizer(
        qa_text,
        return_tensors="pt",
        truncation=(TRUNCATE_MAXLEN is not None),
        max_length=TRUNCATE_MAXLEN,
    )

    input_ids = enc_qa["input_ids"]              # [1, S]
    response_start = enc_user["input_ids"].shape[1]  # int

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
    # 保持在 GPU 上，只做 detach，ICRScore 内部会自己处理设备逻辑
    hs_tuple  = tuple(h.detach() for h in out.hidden_states)
    att_tuple = tuple(a.detach() for a in out.attentions)
    return hs_tuple, att_tuple


def pack_for_icr_from_full_forward(hidden_states_tuple, attentions_tuple, response_start, batch_idx=0):

    L_plus_1 = len(hidden_states_tuple)
    L = L_plus_1 - 1
    B, S, D = hidden_states_tuple[0].shape
    input_len   = response_start
    output_size = max(0, S - input_len)

    # ---- hidden_states ----
    nested_hidden_states = []
    hs_input_per_layer = []
    for ell in range(L_plus_1):
        h = hidden_states_tuple[ell][batch_idx, :input_len, :]   # (input_len, D)
        hs_input_per_layer.append(h.unsqueeze(0))                # (1, input_len, D)
    nested_hidden_states.append(hs_input_per_layer)

    for t in range(output_size):
        pos = input_len + t
        hs_t_per_layer = []
        for ell in range(L_plus_1):
            h = hidden_states_tuple[ell][batch_idx, pos:pos+1, :]  # (1, D)
            hs_t_per_layer.append(h.unsqueeze(0))                  # (1, 1, D)
        nested_hidden_states.append(hs_t_per_layer)

    # ---- attentions ----
    nested_attentions = []
    attn_input_per_layer = []
    for ell in range(L):
        A = attentions_tuple[ell][batch_idx]                      # (H, S, S)
        attn_input_per_layer.append(A.unsqueeze(0))               # (1, H, S, S)
    nested_attentions.append(attn_input_per_layer)

    for t in range(output_size):
        pos = input_len + t
        attn_t_per_layer = []
        for ell in range(L):
            A = attentions_tuple[ell][batch_idx]                  # (H, S, S)
            row = A[:, pos:pos+1, :]                              # (H, 1, S)
            attn_t_per_layer.append(row.unsqueeze(0))             # (1, H, 1, S)
        nested_attentions.append(attn_t_per_layer)

    return nested_hidden_states, nested_attentions


def layer_means_from_icr_scores(icr_scores_item):

    L = len(icr_scores_item)
    means = np.zeros(L, dtype=np.float32)
    for l in range(L):
        vals = icr_scores_item[l]
        means[l] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return means


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
        ds = load_dataset("google-research-datasets/nq_open", split="validation", download_config=download_config)
    else:
        raise ValueError(f"Unknown dataset: {dt}")
    return ds


def _get_question(ds, i):
    return ds[i]["question"]


def _answers_path(dt, model_tag, i):
    return os.path.join(
        ANS_BASE,
        f"{dt}_hal_det/answers_{model_tag}/most_likely_hal_det_{dt}_answers_index_{i}.npy",
    )


def _load_answers(path, max_answers=None):
    """
    简单版 loader：
    - 文件不存在 => []
    - 原样加载 npy 里的对象，不做清洗 / 去重
    - 如指定 max_answers，则截断
    """
    if not os.path.exists(path):
        return []
    arr = np.load(path, allow_pickle=True)
    answers = list(arr)
    if max_answers is not None and len(answers) > max_answers:
        answers = answers[:max_answers]
    # 全部转成 str，防止 bytes 之类
    answers = [a.decode("utf-8", errors="replace") if isinstance(a, bytes) else str(a) for a in answers]
    return answers


def save_dataset_npz(dt, feats, metas, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X = np.stack(feats, axis=0) if len(feats) > 0 else np.zeros((0, 0), dtype=np.float32)
    meta = np.array(metas, dtype=object)
    np.savez_compressed(os.path.join(out_dir, f"{dt}_icr_features.npz"), X=X, meta=meta)


def main():
    # ---- 加载 tokenizer / model ----
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
        torch_dtype=(torch.float16 if DEVICE == "cuda" else torch.float32),
        device_map=("auto" if DEVICE == "cuda" else None),
        token=ACCESS_TOKEN,
    ).eval()
    torch.set_grad_enabled(False)

    os.makedirs(OUT_DIR, exist_ok=True)

    for dt in DATASETS:
        ds = _load_dataset(dt)
        length = len(ds)
        print(f"[{dt}] total samples: {length}")

        feats, metas = [], []

        for i in tqdm(range(length), desc=f"{dt}"):
            question = _get_question(ds, i)
            ans_path = _answers_path(dt, MODEL_TAG, i)
            answers  = _load_answers(ans_path, max_answers=MAX_ANSWERS_PER_QUESTION)
            if not answers:
                continue

            for k, anw in enumerate(answers):
                try:
                    # 1) (Q,A) 编码
                    input_ids, positions = encode_qa(tokenizer, question, anw)

                    # 2) 一次前向
                    hs_tuple, att_tuple = forward_full(model, input_ids, device=DEVICE)

                    # 3) 打包为 ICRScore 期望结构
                    nested_hs, nested_attn = pack_for_icr_from_full_forward(
                        hidden_states_tuple=hs_tuple,
                        attentions_tuple=att_tuple,
                        response_start=positions["response_start"],
                        batch_idx=0,
                    )

                    # 4) 直接计算 ICR，并对回答 token 做 mean pooling → (L,)
                    icr = ICRScore(
                        hidden_states=nested_hs,
                        attentions=nested_attn,
                        core_positions=positions,
                        icr_device=("cuda" if DEVICE == "cuda" else "cpu"),
                        skew_threshold=3,
                        entropy_threshold=3,  # 只在 use_induction_head=True 时启用
                    )
                    icr_scores_item, _ = icr.compute_icr(
                        top_k=TOP_K,
                        top_p=None,
                        pooling="mean",
                        attention_uniform=False,
                        hidden_uniform=False,
                        use_induction_head=USE_INDUCTION_HEAD,
                    )
                    feat = layer_means_from_icr_scores(icr_scores_item)  # (L,)
                    feats.append(feat)
                    metas.append({"q_idx": i, "a_idx": k})

                except RuntimeError as e:
                    print(f"[warn] skip {dt} i={i} k={k}: {repr(e)}")
                finally:
                    # 释放引用，避免显存滞留
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
                    # 不要在每个样本 empty_cache，会很慢；只有 OOM 或大阶段时再手动调用

        # 5) 该数据集全部结束 → 写单个文件
        save_dataset_npz(dt, feats, metas, OUT_DIR)
        print(
            f"[{dt}] saved: {os.path.join(OUT_DIR, f'{dt}_icr_features.npz')} "
            f"(N={len(feats)}, L={feats[0].shape[0] if feats else 0})"
        )

    print(f"All done. Files under: {OUT_DIR}")


if __name__ == "__main__":
    main()
