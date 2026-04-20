# baseline_detectors/detectors/uncertainty_metrics.py
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
        # 声明需要原始生成的对数概率
        self.requires_logprobs = True

    def predict_score(self, accessor) -> float:
        try:
            # 🎯 核心改动 1：优先去拿大管家挂载的独立补票数据！
            logprobs = getattr(accessor, "recovered_logprobs", None)
            
            # 兜底：如果没挂载，才去试老接口
            if logprobs is None: 
                logprobs = accessor.get_token_logprobs()
                
            if logprobs is None or len(logprobs) == 0: 
                return float('nan')
            
            # 🎯 核心改动 2：安全清洗，过滤掉特殊的 None 坏账
            valid_logprobs = [float(p) for p in logprobs if p is not None]
            if not valid_logprobs: 
                return float('nan')
            
            neg_log_likelihood = -np.mean(valid_logprobs)
            
            # 🎯 核心改动 3：防指数爆炸
            if neg_log_likelihood > 50: 
                return float(1e10) 
                
            return float(np.exp(neg_log_likelihood))
        except Exception as e:
            logger.debug(f"[{self.name}] 样本 {accessor.sample_id} 计算失败: {e}")
            return float('nan')


@register_detector("ln_entropy")
class LNEntropyDetector(BaseDetector):
    def __init__(self, name="ln_entropy", **kwargs):
        super().__init__(name, **kwargs)
        self.requires_logprobs = True

    def predict_score(self, accessor) -> float:
        try:
            # 🎯 同理：优先去拿大管家挂载的数据
            logprobs = getattr(accessor, "recovered_logprobs", None)
            
            if logprobs is None: 
                logprobs = accessor.get_token_logprobs()
                
            if logprobs is None or len(logprobs) == 0: 
                return float('nan')
            
            valid_logprobs = [float(p) for p in logprobs if p is not None]
            if not valid_logprobs: 
                return float('nan')
            
            # 长度归一化熵
            ln_entropy = -np.mean(valid_logprobs)
            return float(ln_entropy)
            
        except Exception as e:
            logger.debug(f"[{self.name}] 样本 {accessor.sample_id} 计算失败: {e}")
            return float('nan')