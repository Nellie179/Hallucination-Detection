import torch
import torch.nn.functional as F
import time
import numpy as np


def move_tensors_to_device(container, device):
    if isinstance(container, torch.Tensor):
        return container.to(device)
    elif isinstance(container, list):
        return [move_tensors_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in container]
    elif isinstance(container, tuple):
        return tuple(move_tensors_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in container)
    elif isinstance(container, dict):
        return {k: move_tensors_to_device(v, device) if isinstance(v, torch.Tensor) else v for k, v in
                container.items()}
    else:
        return container


class ICRScore:
    def __init__(self, hidden_states, attentions, skew_threshold=3, entropy_threshold=3, core_positions=None,
                 icr_device=None, use_induction_head=False):
        self.origional_device = hidden_states[0][0].device
        self.icr_device = icr_device
        if self.icr_device != self.origional_device:
            hidden_states = move_tensors_to_device(hidden_states, self.icr_device)
            attentions = move_tensors_to_device(attentions, self.icr_device)

        self.input_lens = hidden_states[0][0].shape[1]
        self.core_positions = core_positions
        with torch.no_grad():
            self.origin_hidden_states = self._pre_process_hs(hidden_states)
            self.origin_attentions = self._pre_process_attn(attentions)

        self.output_hidden_states = self.origin_hidden_states[:, self.input_lens:]
        self.output_attentions = self.origin_attentions[:, :, self.input_lens:]

        self.induction_head = None
        if use_induction_head:
            with torch.no_grad():
                self.induction_head = self._is_induction_head(skew_threshold=skew_threshold,
                                                              entropy_threshold=entropy_threshold)

    def _pre_process_hs(self, hidden_states):
        hidden_states_input = torch.stack(hidden_states[0], dim=0)
        hs_input = hidden_states_input[:, 0, :]
        hidden_states_output = torch.stack([torch.stack(layer) for layer in hidden_states[1:]], dim=0)
        hs_output = torch.cat([hidden_states_output[i, :, 0] for i in range(len(hidden_states_output))], dim=1)
        hs_all = torch.cat([hs_input, hs_output], dim=1)
        del hidden_states_input, hs_input, hidden_states_output, hs_output
        return hs_all

    def _pre_process_attn(self, attentions):
        S = attentions[0][0][0].shape[-1]
        H = attentions[0][0][0].shape[0]
        L = len(attentions[0])

        input_len = self.input_lens
        T_by_seq = max(0, S - input_len)
        T_by_list = max(0, len(attentions) - 1)
        T = min(T_by_seq, T_by_list)
        token_num = input_len + T

        input_attn = torch.stack([attentions[0][layer_idx][0][:, :input_len, :] for layer_idx in range(L)], dim=0)
        pad_size = token_num - input_attn.shape[-1]
        input_attn = F.pad(input_attn, (0, pad_size))

        out_list = []
        for t in range(T):
            t_tensor = torch.stack([attentions[t + 1][layer_idx][0][:, 0, :] for layer_idx in range(L)], dim=0)
            out_list.append(t_tensor.unsqueeze(2))

        if out_list:
            output_attn = torch.cat(out_list, dim=2)
            output_attn = F.pad(output_attn, (0, token_num - output_attn.shape[-1]))
        else:
            output_attn = torch.zeros((L, H, 0, token_num), device=input_attn.device, dtype=input_attn.dtype)

        attn_all_o = torch.cat([input_attn, output_attn], dim=2)
        attn_all = self.set_other_attn_scores_to_zero(attn_all_o)

        del input_attn, output_attn, attn_all_o
        return attn_all

    def set_other_attn_scores_to_zero(self, attn_all):
        layer_num, head_num, seq_len_q, seq_len_k = attn_all.size()
        a = self.core_positions['user_prompt_start']
        b = self.core_positions['user_prompt_end']
        c = self.core_positions['response_start']

        mask = torch.zeros((seq_len_q, seq_len_k), dtype=torch.bool, device=attn_all.device)

        a_safe = min(a, seq_len_q, seq_len_k)
        b_q, b_k = min(b, seq_len_q), min(b, seq_len_k)
        c_q, c_k = min(c, seq_len_q), min(c, seq_len_k)

        mask[a_safe:b_q, a_safe:b_k] = True
        mask[c_q:, c_k:] = True

        attn_all[:, :, ~mask] = 0
        return attn_all

    def _calculate_skewness_entropy(self, attn_map):
        sequence_size = attn_map.size(0)
        row_sums = attn_map.sum(dim=1, keepdim=True)
        row_normalized = attn_map / (row_sums + 1e-12)
        indices = torch.arange(1, sequence_size + 1, device=attn_map.device, dtype=attn_map.dtype).view(1, -1)

        mean_indices = (row_normalized * indices).sum(dim=1)
        variance = ((indices - mean_indices.unsqueeze(1)) ** 2 * row_normalized).sum(dim=1)
        third_moment = ((indices - mean_indices.unsqueeze(1)) ** 3 * row_normalized).sum(dim=1)
        skewness = third_moment / (variance ** 1.5 + 1e-12)
        entropy = -torch.sum(row_normalized * torch.log2(row_normalized + 1e-12), dim=1)

        valid_rows = row_sums.squeeze() > 0
        average_skewness = skewness[valid_rows].mean().item()
        average_entropy = entropy[valid_rows].mean().item()

        return average_skewness, average_entropy

    def _is_induction_head(self, skew_threshold, entropy_threshold):
        is_induction_layer_head = []
        skew_entropy_values = []
        for layer_attentions in self.origin_attentions:
            num_heads = layer_attentions.size(0)
            skewness_entropy = torch.zeros(num_heads, 2, device=layer_attentions.device)

            for head_idx in range(num_heads):
                attn_map = layer_attentions[head_idx]
                skewness, entropy = self._calculate_skewness_entropy(attn_map)
                skewness_entropy[head_idx] = torch.tensor([skewness, entropy])

            skewness = skewness_entropy[:, 0]
            entropy = skewness_entropy[:, 1]
            is_induction_head = (skewness >= skew_threshold) & (entropy <= entropy_threshold)

            if is_induction_head.sum() < num_heads // 8:
                top_heads = skewness.topk(num_heads // 8, largest=True).indices
                is_induction_head[:] = False
                is_induction_head[top_heads] = True

            skew_entropy_values.append(skewness_entropy)
            is_induction_layer_head.append(is_induction_head.tolist())
        return is_induction_layer_head

    def _pooling_attn(self, pooling, use_induction_head):
        pooled_attentions = []
        for layer_idx in range(len(self.output_attentions)):
            induction_heads_this_layer = []
            for head_idx in range(len(self.output_attentions[layer_idx])):
                if use_induction_head and self.induction_head is not None:
                    if self.induction_head[layer_idx][head_idx]:
                        induction_heads_this_layer.append(self.output_attentions[layer_idx][head_idx])
                else:
                    induction_heads_this_layer.append(self.output_attentions[layer_idx][head_idx])

            if induction_heads_this_layer:
                stacked_heads = torch.stack(induction_heads_this_layer)
                if pooling == 'mean':
                    pooled_layer = torch.mean(stacked_heads, dim=0)
                elif pooling == 'max':
                    pooled_layer = torch.max(stacked_heads, dim=0)[0]
                elif pooling == 'min':
                    pooled_layer = torch.min(stacked_heads, dim=0)[0]
                else:
                    raise ValueError(f"{pooling} is not a valid pooling method.")
                pooled_attentions.append(pooled_layer)
            else:
                input_size = self.output_attentions[layer_idx][0].shape[-2:] if self.output_attentions[layer_idx] else (
                1, 1)
                pooled_attentions.append(torch.zeros(input_size))
                raise ValueError(f"Layer {layer_idx} has no induction head.")
        return pooled_attentions

    def compute_icr(self, top_k, top_p, pooling, attention_uniform, hidden_uniform, use_induction_head):
        self.pooling_attentions = self._pooling_attn(pooling=pooling, use_induction_head=use_induction_head)
        icr_scores_item = []

        for layer_idx in range(len(self.pooling_attentions)):
            layer_attn = self.pooling_attentions[layer_idx]
            T_attn, S_attn = layer_attn.shape

            k = min(top_k, S_attn) if top_k is not None else S_attn
            if top_p is not None: k = int(top_p * S_attn)

            attn_topk, attn_topk_idx = torch.topk(layer_attn, k=k, dim=1)

            hs_diff = self.output_hidden_states[layer_idx + 1] - self.output_hidden_states[layer_idx]
            all_hs = self.origin_hidden_states[layer_idx]

            attn_topk_idx = torch.clamp(attn_topk_idx, max=all_hs.shape[0] - 1)
            hs_topk = all_hs[attn_topk_idx]

            min_t = min(hs_diff.shape[0], hs_topk.shape[0], attn_topk.shape[0])
            if min_t == 0:
                icr_scores_item.append([])
                continue

            hs_diff = hs_diff[:min_t]
            hs_topk = hs_topk[:min_t]
            attn_topk = attn_topk[:min_t]

            w_i = torch.einsum('th,tkh->tk', hs_diff, hs_topk)
            w_i = w_i / (torch.norm(hs_topk, dim=2) + 1e-8)

            if attention_uniform:
                attn_topk = torch.ones_like(attn_topk) / k
            if hidden_uniform:
                w_i = torch.ones_like(w_i) / k

            layer_scores = js_divergence_batch(w_i, attn_topk)
            icr_scores_item.append(layer_scores.cpu().tolist())

        return icr_scores_item, k / max(S_attn, 1e-6)


def js_divergence_batch(p, q):
    p_mean = p.mean(dim=1, keepdim=True)
    p_std = p.std(dim=1, keepdim=True).clamp(min=1e-8)
    p = (p - p_mean) / p_std

    q_mean = q.mean(dim=1, keepdim=True)
    q_std = q.std(dim=1, keepdim=True).clamp(min=1e-8)
    q = (q - q_mean) / q_std

    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)

    m = 0.5 * (p + q)

    kl_pm = (p * (p / (m + 1e-12)).log()).sum(dim=1)
    kl_qm = (q * (q / (m + 1e-12)).log()).sum(dim=1)

    return 0.5 * kl_pm + 0.5 * kl_qm