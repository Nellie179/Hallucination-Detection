import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Any
# 对齐你的主干依赖
from baseline_detectors.data_utils.llm_layers import add_tsv_layers, get_layers
from baseline_detectors.data_utils.train_utils import get_last_non_padded_token_rep, compute_ot_loss_cos, update_centroids_ema_hard

def collate_fn(prompts, labels, pad_token_id):
    max_seq_len = max(prompt.shape[1] for prompt in prompts)
    batch_size = len(prompts)
    # 使用 float32 的 pad_token_id 容器，input_ids 保持 long
    prompts_padded = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        seq_len = prompt.shape[1]
        prompts_padded[i, :seq_len] = prompt.squeeze(0)
    labels = torch.tensor(labels, dtype=torch.long)
    return prompts_padded, labels

class TSVFeatureExtractor:
    """
    符合 Benchmark 规范的 TSV 特征提取器。
    🚀 [全监督版]: 直接利用 LLM Judge 的 eval_result 进行训练。
    """
    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None):
        if model is None or tokenizer is None:
            raise ValueError("❌ [TSVExtractor] 必须从 runner 传入共享的 model 和 tokenizer 实例！")
        
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        print(f"[*] TSVFeatureExtractor 已接管模型，当前推理精度: {self.model.dtype}")

    def _prepare_data(self, input_jsonl_path, num_train_samples):
        """解析 03 文件，对齐 eval_result 标签"""
        prompts, labels, sample_ids = [], [], []
        
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # 🚀 关键：根据你的 03 文件样例，Key 是 eval_result
                judge_val = str(item.get("eval_result", "correct")).lower()
                # label 1 为正确，0 为幻觉
                label = 1 if "correct" in judge_val else 0
                
                # 拿取已经渲染好的 full prompt 和模型输出
                full_prompt = item.get("prompt", "")
                model_ans = item.get("model_output_text", "")
                
                if not full_prompt or not model_ans: continue

                # 完美拼接：Llama-3 的完整对话流
                # 假设 full_prompt 已经包含了 assistant header，我们补上答案和 eot
                full_text = f"{full_prompt}{model_ans}{self.tokenizer.eos_token}"
                
                prompt_ids = self.tokenizer(full_text, return_tensors='pt').input_ids.to(self.device)
                
                prompts.append(prompt_ids)
                labels.append(label)
                sample_ids.append(item["sample_id"])

        # 随机抽取一部分做全监督训练，剩下的做测试
        train_prompts = prompts[:num_train_samples]
        train_labels = labels[:num_train_samples]
        return train_prompts, train_labels, prompts, labels, sample_ids

    def process_and_extract(
        self,
        input_jsonl_path: str,
        output_jsonl_path: str,
        num_train_samples: int = 500,
        epochs: int = 40,
        batch_size: int = 8,
        lr: float = 0.005,
        str_layer: int = 9,
        lam: float = 5.0
    ):
        print(f"[*] >>> 启动 TSV 任务引导向量训练管线 (层: {str_layer})")

        train_p, train_l, all_p, all_l, all_ids = self._prepare_data(input_jsonl_path, num_train_samples)
        
        # --- 阶段 1: 注入高精度 TSV 参数 ---
        # 冻结底座，保存原状态
        original_requires_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters(): p.requires_grad = False
            
        hidden_size = self.model.config.hidden_size
        # 🚀 精度防线：参数全部强制 float32
        tsv_params = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32), requires_grad=True) 
            for _ in range(self.model.config.num_hidden_layers)
        ]).to(self.device)
        
        class DummyArgs: pass
        args = DummyArgs()
        args.component, args.str_layer, args.model_name = 'res', str_layer, self.model_name
        args.cos_temp, args.ema_decay = 0.1, 0.99
        
        # 注入残差流劫持层
        add_tsv_layers(self.model, tsv_params, [lam], args)
        optimizer = torch.optim.AdamW(list(tsv_params.parameters()), lr=lr)

        # 质心初始化 (float32)
        centroids = F.normalize(torch.randn((2, hidden_size), dtype=torch.float32).to(self.device), p=2, dim=1)

        self.model.eval()
        # =========================================================
        # 🚀 钩子 1：引燃火种，强行从 Embedding 赋予梯度，逼迫 PyTorch 建图
        # =========================================================
        def force_grad_hook(module, inp, out):
            out.requires_grad_(True)
        emb_hook = self.model.get_input_embeddings().register_forward_hook(force_grad_hook)

        # =========================================================
        # 🚀 钩子 2：窃听器，直接抓取 TSV 层输出，彻底绕开多卡 Detach 黑洞
        # =========================================================
        tracked_hiddens = []
        def intercept_hook(module, input, output):
            tracked_hiddens.append(output)
            
        layers = get_layers(self.model)
        hook_handle = layers[str_layer].tsv_layer.register_forward_hook(intercept_hook)

        # =========================================================
        # 🚀 终极破局点：强制开启梯度引擎！冲破 runner.py 的全局封锁
        # =========================================================
        with torch.enable_grad():
            for epoch in range(epochs):
                indices = torch.randperm(len(train_p)).tolist()
                cur_p = [train_p[idx] for idx in indices]
                cur_l = [train_l[idx] for idx in indices]

                pbar = tqdm(range(0, len(cur_p), batch_size), desc=f"TSV Train Ep {epoch+1}")
                for start in pbar:
                    batch_p, batch_l = cur_p[start:start+batch_size], cur_l[start:start+batch_size]
                    b_in, b_labels_t = collate_fn(batch_p, batch_l, self.tokenizer.pad_token_id)
                    b_in = b_in.to(self.device)
                    
                    # 每次前向传播前，清空窃听口袋
                    tracked_hiddens.clear()
                    
                    # ⚠️ 必须关闭 output_hidden_states，切断底层原生 Tuple 收集以防断流！
                    _ = self.model(b_in, output_hidden_states=False)
                    
                    # 🚀 直接从 Hook 口袋里拿出绝对带有梯度的隐藏层张量
                    target_hidden = tracked_hiddens[0]
                    
                    # 🚀 修复点 2：将掩码对齐到目标隐藏层所在的具体显卡 (防御 H200 跨卡报错)
                    attn_mask = (b_in != self.tokenizer.pad_token_id).to(target_hidden.device)
                    last_token_rep = get_last_non_padded_token_rep(target_hidden, attn_mask).to(torch.float32)
                    
                    # 🚀 修复点 3：算 Loss 前将其他变量跨卡运输对齐
                    centroids_dev = centroids.to(target_hidden.device)
                    b_labels_oh = F.one_hot(b_labels_t.to(target_hidden.device), num_classes=2).to(torch.float32)
                    
                    ot_loss, _ = compute_ot_loss_cos(last_token_rep, centroids_dev, b_labels_oh, len(batch_p), args)
                    
                    ot_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        # 更新全局质心时，带回主卡
                        centroids = update_centroids_ema_hard(centroids, last_token_rep.to(self.device), b_labels_oh.to(self.device), args)
                    pbar.set_postfix({"loss": f"{ot_loss.item():.4f}"})

        # 🚀 训练结束，立刻拆除所有钩子，还原大模型清白之身
        emb_hook.remove()
        hook_handle.remove()
        # --- 阶段 2: 全量推理并存分 ---
        print("[*] 训练完成，开始全自动特征落盘...")
        for p in tsv_params.parameters(): p.requires_grad = False
        
        # --- 阶段 2: 全量推理并存分 ---
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            with torch.no_grad():
                # 🚀 修正了切片中的 batch_size 变量名
                for i in tqdm(range(0, len(all_p), batch_size), desc="TSV Prediction"):
                    batch_p = all_p[i : i + batch_size]
                    batch_ids = all_ids[i : i + batch_size]
                    
                    b_in, _ = collate_fn(batch_p, [0] * len(batch_p), self.tokenizer.pad_token_id)
                    b_in = b_in.to(self.device)
                    
                    output = self.model(b_in, output_hidden_states=True)
                    target_hidden = torch.stack(output.hidden_states, dim=0).squeeze()[str_layer]
                    last_token_rep = get_last_non_padded_token_rep(target_hidden, (b_in != self.tokenizer.pad_token_id)).to(torch.float32)
                    
                    last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                    sims = torch.matmul(last_token_rep, F.normalize(centroids, p=2, dim=-1).T) / args.cos_temp
                    hallu_probs = torch.softmax(sims, dim=-1)[:, 0].cpu().numpy().tolist()
                    
                    for idx, sid in enumerate(batch_ids):
                        f_out.write(json.dumps({"sample_id": sid, "tsv_hallucination_score": hallu_probs[idx]}) + "\n")                        
        # --- 阶段 3: 还原现场 ---
        layers = get_layers(self.model)
        if hasattr(layers[str_layer], 'llama_decoder_layer'):
            layers[str_layer] = layers[str_layer].llama_decoder_layer # 拆掉 Wrapper
            
        for n, p in self.model.named_parameters():
            if n in original_requires_grad: p.requires_grad = original_requires_grad[n]
        
        torch.cuda.empty_cache()
        print(f"[+] TSV 组件执行完毕，分数已保存至: {output_jsonl_path}")