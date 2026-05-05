import numpy as np
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("perplexity")
class PerplexityDetector(BaseDetector):
    def __init__(self, name="perplexity", **kwargs):
        super().__init__(name, **kwargs)
        # Perplexity 是评估单次生成质量的，只需要基础对数概率
        self.requires_logprobs = True

    def predict_score(self, accessor) -> float:
        try:
            # 优先用补票数据，兜底用原始生成数据
            logprobs = getattr(accessor, "recovered_logprobs", None)
            if logprobs is None: 
                logprobs = accessor.get_token_logprobs()
                
            if logprobs is None or len(logprobs) == 0: 
                return float('nan')
            
            # 过滤异常值
            valid_logprobs = [float(p) for p in logprobs if p is not None and not np.isnan(float(p))]
            if not valid_logprobs: 
                return float('nan')
            
            neg_log_likelihood = -np.mean(valid_logprobs)
            
            # 防指数爆炸
            if neg_log_likelihood > 50: 
                return float(1e10) 
                
            return float(np.exp(neg_log_likelihood))
        except Exception as e:
            logger.debug(f"[{self.name}] 样本 {accessor.sample_id} 计算 PPL 失败: {e}")
            return float('nan')


@register_detector("ln_entropy")
class LNEntropyDetector(BaseDetector):
    def __init__(self, name="ln_entropy", **kwargs):
        super().__init__(name, **kwargs)
        # 声明需要多次采样的序列概率
        self.requires_stochastic = True
        self.requires_logprobs = True # 用于兜底

    def predict_score(self, accessor) -> float:
        try:
            # 🚀 1. 直接调用 accessor 的原生接口，获取一维 float 列表
            st_logprobs = accessor.get_stochastic_logprobs()
            
            if st_logprobs and len(st_logprobs) > 0:
                # 这些值已经是序列级的 log likelihood
                valid_st_lps = [float(p) for p in st_logprobs if p is not None and not np.isnan(float(p))]
                
                if len(valid_st_lps) > 0:
                    # 论文公式: Predictive Entropy ≈ - (1/K) * Σ (Sequence Log Likelihood)
                    # 因为这里直接提供的是 sequence likelihood，直接求均值后取负号即可
                    expected_ln_entropy = -np.mean(valid_st_lps)
                    return float(expected_ln_entropy)

            # 🚀 2. 兜底策略：如果因为某些原因没有随机采样，退化为基础的单次 LN-NLL
            base_lp = getattr(accessor, "recovered_logprobs", None)
            if base_lp is None:
                base_lp = accessor.get_token_logprobs()
                
            if base_lp is not None and len(base_lp) > 0:
                valid_base_lps = [float(p) for p in base_lp if p is not None and not np.isnan(float(p))]
                if len(valid_base_lps) > 0:
                    return float(-np.mean(valid_base_lps))
            
            return float('nan')
            
        except Exception as e:
            logger.debug(f"[{self.name}] 样本 {accessor.sample_id} 计算 Entropy 失败: {e}")
            return float('nan')