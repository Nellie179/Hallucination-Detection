# baseline_detectors/detectors/self_evaluator.py
import re
import logging
import math
import numpy as np
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("self_evaluator")
class SelfEvaluatorDetector(BaseDetector):
    def __init__(self, name="self_evaluator", **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = False
        self.requires_qa_features = True 
        self.requires_logprobs = True

    def _get_text_label(self, text: str) -> str:
        """识别模型到底倾向于哪个结论"""
        if not text: return "neutral"
        text = text.lower()
        neg_patterns = [r"final grade:\s*incorrect", r"is incorrect", r"\bincorrect\b", r"wrong", r"\bfalse\b", r"\bno\b"]
        pos_patterns = [r"final grade:\s*correct", r"is correct", r"\bcorrect\b", r"accurate", r"\btrue\b", r"\byes\b"]
        
        for p in neg_patterns:
            if re.search(p, text): return "incorrect"
        for p in pos_patterns:
            if re.search(p, text): return "correct"
        return "neutral"

    def predict_score(self, accessor) -> float:
        """
        🚀 根源修复：撕掉遮羞布，绝对优先读取底层计算好的 P(True)！
        """
        # =================================================================
        # 1. 绝对优先：从 accessor 读取 H5 里的精准对数概率 (X光片证明它是完好的！)
        # =================================================================
        logprobs = accessor.get_token_logprobs()
        if logprobs and len(logprobs) > 0:
            lp = float(logprobs[0])
            if not (math.isnan(lp) or math.isinf(lp)):
                # lp 是 log(P(True))，因此 exp(lp) 就是最精准的 P(True) 概率
                p_true = np.exp(lp)
                p_true = min(max(p_true, 0.0), 1.0) # 截断保护
                
                # 幻觉分数 = 1.0 - P(True) (如果 P(True) 是 1，说明判定正确，幻觉分是 0)
                return float(1.0 - p_true)

        # =================================================================
        # 2. 降级方案：如果上面抛出异常或没拿到概率，再退化到文本匹配
        # =================================================================
        raw_text = accessor.metadata.get("self_evaluator_raw", "")
        label = self._get_text_label(raw_text)
        
        if label == "correct":
            return 0.0
        elif label == "incorrect":
            return 1.0
            
        # =================================================================
        # 3. 🚨 绝不再悄悄给 0.5 掩耳盗铃！拿不到数据直接让程序崩溃！
        # =================================================================
        raise RuntimeError(f"样本 {accessor.sample_id}: H5 中无对数概率，且正则解析文本失败 ({raw_text})")