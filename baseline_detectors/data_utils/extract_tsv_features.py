import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Any

from baseline_detectors.data_utils.llm_layers import add_tsv_layers, get_layers
from baseline_detectors.data_utils.train_utils import get_last_non_padded_token_rep, compute_ot_loss_cos, update_centroids_ema_hard

def collate_fn(prompts, labels, pad_token_id):
    max_seq_len = max(prompt.shape[1] for prompt in prompts)
    batch_size = len(prompts)
    prompts_padded = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        seq_len = prompt.shape[1]
        prompts_padded[i, :seq_len] = prompt.squeeze(0)
    labels = torch.tensor(labels, dtype=torch.long)
    return prompts_padded, labels

class TSVFeatureExtractor:
    """
    符合 Benchmark 规范的 TSV 特征提取器。
    🚀 [真·全监督版]: 严格拆分 Train 与 Eval，支持训练 Vector 拔插。
    """
    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None):
        if model is None or tokenizer is None:
            raise ValueError("❌ [TSVExtractor] 必须从 runner 传入共享的 model 和 tokenizer 实例！")
        
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        print(f"[*] TSVFeatureExtractor 已接管模型，当前推理精度: {self.model.dtype}")

    def _prepare_data_simple(self, jsonl_path):
        prompts, labels, sample_ids = [], [], []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # 读取切分数据集时的 eval_category 标签
                judge_val = str(item.get("eval_category", "correct")).lower()
                label = 1 if "correct" in judge_val else 0
                
                full_prompt = item.get("prompt", "")
                model_ans = item.get("model_output_text", "")
                if not full_prompt or not model_ans: continue

                full_text = f"{full_prompt}{model_ans}{self.tokenizer.eos_token}"
                prompt_ids = self.tokenizer(full_text, return_tensors='pt').input_ids.to(self.device)
                
                prompts.append(prompt_ids)
                labels.append(label)
                sample_ids.append(item["sample_id"])
        return prompts, labels, sample_ids

    def train_vector(self, train_jsonl_path, str_layer, epochs=20, batch_size=8, lr=0.005, lam=5.0):
        """
        🚀 纯训练函数：只在 Train 集上训练，返回训练好的参数，不产生评测文件。
        (极致省显存版：切断全局梯度建图)
        """
        train_p, train_l, _ = self._prepare_data_simple(train_jsonl_path)
        
        original_requires_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters(): p.requires_grad = False
            
        hidden_size = self.model.config.hidden_size
        tsv_params = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32), requires_grad=True) 
            for _ in range(self.model.config.num_hidden_layers)
        ]).to(self.device)
        
        class DummyArgs: pass
        args = DummyArgs()
        args.component, args.str_layer, args.model_name = 'res', str_layer, self.model_name
        args.cos_temp, args.ema_decay = 0.1, 0.99
        
        add_tsv_layers(self.model, tsv_params, [lam], args)
        optimizer = torch.optim.AdamW(list(tsv_params.parameters()), lr=lr)

        centroids = F.normalize(torch.randn((2, hidden_size), dtype=torch.float32).to(self.device), p=2, dim=1)

        self.model.eval()
        
        # =========================================================
        # 🚀 钩子：窃听器与梯度物理断电闸门
        # 移除了致命的 emb_hook，不再强迫大模型全局通电建图
        # =========================================================
        tracked_hiddens = []
        def intercept_hook(module, input, output): 
            hidden = output[0] if isinstance(output, tuple) else output
            
            # 1. 抓取真实带梯度的张量存进口袋
            tracked_hiddens.append(hidden)
            
            # 2. 🚀 关键修改：将流向后续层的张量强行 detach，后续所有层变成纯推理，暴省 80% 显存
            detached_hidden = hidden.detach()
            
            if isinstance(output, tuple):
                return (detached_hidden,) + output[1:]
            return detached_hidden
            
        layers = get_layers(self.model)
        # 注意这里直接挂载在 layers[str_layer] 上
        hook_handle = layers[str_layer].register_forward_hook(intercept_hook)

        with torch.enable_grad():
            for epoch in range(epochs):
                indices = torch.randperm(len(train_p)).tolist()
                cur_p = [train_p[idx] for idx in indices]
                cur_l = [train_l[idx] for idx in indices]

                pbar = tqdm(range(0, len(cur_p), batch_size), desc=f"Train TSV (Layer {str_layer}) Ep {epoch+1}", leave=False)
                for start in pbar:
                    batch_p, batch_l = cur_p[start:start+batch_size], cur_l[start:start+batch_size]
                    b_in, b_labels_t = collate_fn(batch_p, batch_l, self.tokenizer.pad_token_id)
                    b_in = b_in.to(self.device)
                    
                    tracked_hiddens.clear()
                    _ = self.model(b_in, output_hidden_states=False)
                    
                    # 🚀 从口袋里拿出真正带梯度的截断层张量算 Loss
                    target_hidden = tracked_hiddens[0]
                    attn_mask = (b_in != self.tokenizer.pad_token_id).to(target_hidden.device)
                    last_token_rep = get_last_non_padded_token_rep(target_hidden, attn_mask).to(torch.float32)
                    
                    centroids_dev = centroids.to(target_hidden.device)
                    b_labels_oh = F.one_hot(b_labels_t.to(target_hidden.device), num_classes=2).to(torch.float32)
                    
                    ot_loss, _ = compute_ot_loss_cos(last_token_rep, centroids_dev, b_labels_oh, len(batch_p), args)
                    
                    ot_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        centroids = update_centroids_ema_hard(centroids, last_token_rep.to(self.device), b_labels_oh.to(self.device), args)
                    pbar.set_postfix({"loss": f"{ot_loss.item():.4f}"})

        hook_handle.remove()
        
        # 🚀 提取训练好的物理张量
        trained_vector = tsv_params[str_layer].detach().clone()
        final_centroids = centroids.detach().clone()
        
        # 还原大模型现场，拔掉 Wrapper
        if hasattr(layers[str_layer], 'llama_decoder_layer'):
            layers[str_layer] = layers[str_layer].llama_decoder_layer 
            
        for n, p in self.model.named_parameters():
            if n in original_requires_grad: p.requires_grad = original_requires_grad[n]
            
        return trained_vector, final_centroids

    def evaluate_vector(self, eval_jsonl_path, output_jsonl_path, trained_vector, final_centroids, str_layer, batch_size=8, lam=5.0):
        """
        🚀 纯推理函数：接收训练好的 Vector 和质心，插入模型，对目标集合进行打分并落盘。
        """
        eval_p, eval_l, eval_ids = self._prepare_data_simple(eval_jsonl_path)
        
        hidden_size = self.model.config.hidden_size
        tsv_params = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32), requires_grad=False) 
            for _ in range(self.model.config.num_hidden_layers)
        ]).to(self.device)
        
        # 🔌 将传入的“最强外挂”插在指定的层上
        tsv_params[str_layer].data.copy_(trained_vector)
        
        class DummyArgs: pass
        args = DummyArgs()
        args.component, args.str_layer, args.model_name = 'res', str_layer, self.model_name
        args.cos_temp = 0.1
        
        add_tsv_layers(self.model, tsv_params, [lam], args)
        
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            with torch.no_grad():
                for i in tqdm(range(0, len(eval_p), batch_size), desc=f"Eval TSV (Layer {str_layer})", leave=False):
                    batch_p = eval_p[i : i + batch_size]
                    batch_ids = eval_ids[i : i + batch_size]
                    
                    b_in, _ = collate_fn(batch_p, [0] * len(batch_p), self.tokenizer.pad_token_id)
                    b_in = b_in.to(self.device)
                    
                    output = self.model(b_in, output_hidden_states=True)
                    # 修复了 squeeze 的 bug，直接通过索引提取，速度更快更安全
                    target_hidden = output.hidden_states[str_layer + 1] 
                    
                    attn_mask = (b_in != self.tokenizer.pad_token_id).to(target_hidden.device)
                    last_token_rep = get_last_non_padded_token_rep(target_hidden, attn_mask).to(torch.float32)
                    
                    last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                    sims = torch.matmul(last_token_rep, F.normalize(final_centroids, p=2, dim=-1).T) / args.cos_temp
                    hallu_probs = torch.softmax(sims, dim=-1)[:, 0].cpu().numpy().tolist()
                    
                    for idx, sid in enumerate(batch_ids):
                        f_out.write(json.dumps({"sample_id": sid, "tsv_hallucination_score": hallu_probs[idx]}) + "\n")                        
        
        # 用完后拔掉外挂
        layers = get_layers(self.model)
        if hasattr(layers[str_layer], 'llama_decoder_layer'):
            layers[str_layer] = layers[str_layer].llama_decoder_layer
        torch.cuda.empty_cache()