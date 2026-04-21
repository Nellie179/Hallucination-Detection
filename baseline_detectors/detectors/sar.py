# baseline_detectors/detectors/sar.py
"""
SAR (Shifting Attention to Relevance) Detector - 终极安全版

原理：
    结合大模型生成的不确定性与语义相关性，重新加权生成的不确定性分数。

工程优化：
    1. Sentence-SAR 范式：利用大管家缓存的序列级概率。
    2. NLI 矩阵并行化：大幅提速。
    3. 数学级防御：引入 LogSumExp 平移，彻底封死无穷大 (inf) 崩溃。
"""

import os
import math
import torch
import numpy as np
import logging
from typing import List

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logger = logging.getLogger(__name__)

# 尝试导入 sentence_transformers
try:
    from sentence_transformers.cross_encoder import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("未安装 sentence_transformers，SAR 探针将无法工作。")


@register_detector("sar")
class SARDetector(BaseDetector):
    def __init__(
        self, 
        name="sar", 
        measurement_model: str = "cross-encoder/stsb-distilroberta-base",
        t: float = 0.001,
        device: str = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.t = t
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SAR 必须依赖 sentence_transformers。请执行: pip install sentence_transformers")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[{self.name}] 加载 NLI 模型: {measurement_model} 到 {self.device}")
        self.measure_model = CrossEncoder(model_name=measurement_model, device=self.device, num_labels=1)

    def _semantic_weighted_log(self, similarities: List[List[float]], entropies: torch.Tensor) -> torch.Tensor:
        """
        [数学防御核心] LogSumExp 平移技巧，彻底杜绝 exp() 下溢引发的 log(0)=inf 报错
        """
        # entropies 是正数（因为负对数概率取反了），我们需要还原成负数对数概率
        log_probs = -1 * entropies
        
        # 提取最大对数概率作为平移基准，防止集体下溢
        max_log_prob = log_probs.max()
        if torch.isinf(max_log_prob):
            return torch.zeros_like(entropies)
            
        shifted_log_probs = log_probs - max_log_prob
        shifted_probs = torch.exp(shifted_log_probs)
        
        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(shifted_probs, entropies)):
            sim_tensor = torch.tensor(similarities[idx], device=self.device)
            # 强力清洗：消除 NLI 模型偶尔吐出的诡异 NaN/Inf
            sim_tensor = torch.nan_to_num(sim_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            other_probs = torch.cat([shifted_probs[:idx], shifted_probs[idx + 1:]])
            
            # SAR 核心计算
            sum_term = prob + ((sim_tensor / self.t) * other_probs).sum()
            
            # 强力钳制：底线保护，绝不给 log() 喂 <= 0 的数字
            sum_term = torch.clamp(sum_term, min=1e-10)
            
            # 还原平移公式
            w_ent = -(torch.log(sum_term) + max_log_prob)
            weighted_entropy.append(w_ent)
            
        return torch.tensor(weighted_entropy, device=self.device)

    def predict_score(self, accessor: SampleAccessor) -> float:
        # 【临时移除 try-except，让报错直接把程序炸停，我们看 trace！】
        prompt = accessor.get_prompt_text()
        raw_samples = accessor.get_stochastic_samples()
        raw_logprobs = accessor.get_stochastic_logprobs()

        # Debug 1: 检查大管家到底有没有给你送来数据
        if not raw_samples or not raw_logprobs:
            logger.error(f"[SAR 致命拦截] 样本 {accessor.sample_id} 根本没有采样数据或概率！")
            logger.error(f"  - raw_samples 长度: {len(raw_samples) if raw_samples else 'None'}")
            logger.error(f"  - raw_logprobs 长度: {len(raw_logprobs) if raw_logprobs else 'None'}")
            return 0.5
            
        samples, logprobs = [], []
        for s, lp in zip(raw_samples, raw_logprobs):
            if lp is not None and not math.isnan(lp) and not math.isinf(lp):
                samples.append(s)
                logprobs.append(float(lp))
                
        num_generations = len(samples)
        if num_generations <= 1:
            logger.error(f"[SAR 致命拦截] 样本 {accessor.sample_id} 有效采样数不足 2 个，无法计算。")
            return 0.5 

        # Debug 2: 确认接下来进入张量计算阶段
        print(f"[*] 样本 {accessor.sample_id} 数据校验通过 (Generations: {num_generations})，准备执行张量计算...")

        gen_entropies = torch.tensor([-lp for lp in logprobs], dtype=torch.float32, device=self.device)

        pairs = []
        pair_indices = []
        for i in range(num_generations):
            for j in range(i + 1, num_generations):
                pairs.append([prompt + samples[i], prompt + samples[j]])
                pair_indices.append((i, j))

        # 这里如果报错，通常是 CrossEncoder 遇到超长文本 (大于 512 token) 或 OOM
        flat_similarities = self.measure_model.predict(pairs, show_progress_bar=False)

        similarities = {i: [] for i in range(num_generations)}
        for (i, j), sim_score in zip(pair_indices, flat_similarities):
            similarities[i].append(float(sim_score))
            similarities[j].append(float(sim_score))

        # 这里如果报错，说明自定义的数学函数张量维度不匹配
        sar_scores = self._semantic_weighted_log(similarities, gen_entropies)
        final_score = float(sar_scores.mean().cpu().item())
        
        return final_score