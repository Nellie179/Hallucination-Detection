# baseline_detectors/detectors/self_evaluator.py
import re
import logging
import numpy as np
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("self_evaluator")
class SelfEvaluatorDetector(BaseDetector):
    def __init__(self, name="self_evaluator", **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = False
        # 🚨 [关键新增]: 告诉 Runner，我需要像 PRISM/CCS 那样进行特征提取
        self.requires_qa_features = True 
        self.requires_logprobs = True

    def _get_text_label(self, text: str) -> str:
        """识别模型到底倾向于哪个结论"""
        if not text: return "neutral"
        text = text.lower()
        # 预定义正负模式
        neg_patterns = [r"final grade:\s*incorrect", r"is incorrect", r"\bincorrect\b", r"wrong"]
        pos_patterns = [r"final grade:\s*correct", r"is correct", r"\bcorrect\b", r"accurate"]
        
        for p in neg_patterns:
            if re.search(p, text): return "incorrect"
        for p in pos_patterns:
            if re.search(p, text): return "correct"
        return "neutral"

    def predict_score(self, accessor) -> float:
        """
        🚀 真正的 P(True) 逻辑：
        结合生成的文本结论 (Label) 和生成该结论时的置信度 (Logprob)
        """
        # 1. 获取自评文本
        raw_text = accessor.metadata.get("self_evaluator_raw", "")
        label = self._get_text_label(raw_text)
        
        # 2. 尝试获取 Logprobs (信心值)
        # 注意：accessor 会根据 self.name 去 H5 里找对应的 logprobs
        logprobs = accessor.get_token_logprobs()
        
        # 计算首个 Token 的概率 (作为信心值的代理)
        confidence = 1.0 # 默认全信
        if logprobs and len(logprobs) > 0:
            # 提取第一个 Token 的概率: P = exp(log_prob)
            # 大模型通常在输出第一个词(Correct/Incorrect)时最能反映置信度
            confidence = np.exp(float(logprobs[0]))
            confidence = min(max(confidence, 0.0), 1.0) # 截断

        # 3. 映射为连续的幻觉分数 [0.0, 1.0]
        # 💡 逻辑：
        # - 确定的 Correct: 分数 -> 0.0
        # - 犹豫的 Correct/Incorrect: 分数 -> 0.5
        # - 确定的 Incorrect: 分数 -> 1.0
        if label == "correct":
            # 如果模型说对，但信心只有 0.6，分数就是 0.4
            return float(1.0 - confidence)
        elif label == "incorrect":
            # 如果模型说错，且信心高达 0.9，分数就是 0.9
            return float(confidence)
        else:
            # 模棱两可的情况
            return 0.5