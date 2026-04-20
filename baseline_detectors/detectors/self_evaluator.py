import re
import logging
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("self_evaluator")
class SelfEvaluatorDetector(BaseDetector):
    def __init__(self, name="self_evaluator", **kwargs):
        super().__init__(name, **kwargs)
        # 记录该探测器不需要随机采样文本，因为它只看模型对主答案的评价
        self.requires_stochastic = False

    def _parse_label(self, text: str) -> float:
        if not text:
            return 0.5
        
        text = text.lower()
        
        # 🚨 放宽了正则限制，只要包含独立的 correct/incorrect 就抓取！
        negative_patterns = [
            r"final grade:\s*incorrect", 
            r"is incorrect", 
            r"factual error", 
            r"contains hallucination",
            r"wrong",
            r"\bincorrect\b"  # 👈 新增：直接抓取单独的 incorrect
        ]
        positive_patterns = [
            r"final grade:\s*correct", 
            r"is correct", 
            r"is accurate", 
            r"no errors",
            r"\bcorrect\b"    # 👈 新增：直接抓取单独的 correct (完美匹配 " Correct")
        ]

        for pattern in negative_patterns:
            if re.search(pattern, text):
                return 1.0  # 判定为幻觉
        
        for pattern in positive_patterns:
            if re.search(pattern, text):
                return 0.0  # 判定为正确
        
        return 0.5

    def predict_score(self, accessor) -> float:
        """
        从 Accessor 的 metadata 中读取由 Runner 事先生成的 self_evaluator_raw。
        """
        # 获取模型生成的自评字符串（包含 Reasoning）
        raw_eval_text = accessor.metadata.get("self_evaluator_raw", None)
        
        if raw_eval_text is None:
            # 如果没找到，说明 Runner 阶段没有跑自评逻辑
            return float('nan')
            
        # 解析文本得到分数
        score = self._parse_label(raw_eval_text)
        return float(score)