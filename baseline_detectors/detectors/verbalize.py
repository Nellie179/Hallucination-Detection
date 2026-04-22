# baseline_detectors/detectors/verbalize.py
"""
Verbalize Detector - 语言化不确定性检测器

原理：
    统计模型输出中的不确定性表达（如 "I'm not sure", "maybe", "probably"）。
    如果模型频繁使用这些词汇，说明其对答案不确定，可能是幻觉。

特点：
    - 免训练（training-free）
    - 无需额外数据
    - 轻量级，速度最快

适用场景：
    - 所有任务类型
    - 特别适合英文生成任务

参考文献：
    - "Can LLMs Express Their Uncertainty?" (2023)
    - Verbalized confidence calibration研究
"""

# /home/zfang1/Data/Lxy/Benchmark/baseline_detectors/detectors/verbalize.py
import re
import logging
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("verbalize")
class VerbalizeDetector(BaseDetector):
    def __init__(self, name="verbalize", **kwargs):
        super().__init__(name, **kwargs)
        # 参考源码中的 Prompt 核心逻辑：要求模型输出置信度数字
        self.prompt_template = (
            "Answer the following question and then provide a confidence score "
            "between 0 and 1 representing how sure you are about your answer.\n"
            "Question: {question}\n"
            "Answer: {answer}\n"
            "Confidence Score (0-1):"
        )

    def _extract_confidence(self, text: str) -> float:
        """
        🚀 终极修复版正则：精准定位，防误杀，强制截断
        """
        if not text: return 0.5
        text = str(text).lower()
        
        # 1. 优先尝试提取具有明确指示符的分数 (防误抓文本里的数字)
        # 匹配 "confidence: 0.8", "score is 90%" 等
        match = re.search(r'(?:confidence|score|certainty).*?([0-9]*\.?[0-9]+%?)', text)
        
        if match:
            num_str = match.group(1)
        else:
            # 2. 如果没有指示符，只抓取孤立的数字 (排除年份、普通数量词)
            # 使用 \b 确保数字是独立的
            matches = re.findall(r"\b(\d+%\.?\d*|0?\.\d+|1\.0|1|0)\b", text)
            if not matches: return 0.5
            # 如果有多个，假设模型倾向于把最终得分放在末尾
            num_str = matches[-1] 

        # 3. 安全转换与截断
        try:
            if "%" in num_str:
                val = float(num_str.replace("%", "")) / 100.0
            else:
                val = float(num_str)
                # 兼容模型直接输出 1-10 甚至 1-100 的评分制
                if val > 1.0 and val <= 10.0:
                    val = val / 10.0
                elif val > 10.0 and val <= 100.0:
                    val = val / 100.0
                    
            # 🚨 终极保护：确保概率绝对在 0 到 1 之间
            return min(max(val, 0.0), 1.0)
        except Exception:
            return 0.5

    def predict_score(self, accessor) -> float:
        raw_response = accessor.metadata.get("verbalize_response", None)
        
        if not raw_response:
            return float('nan')
            
        # 剔除 prompt 尾巴带来的污染（如果有的话）
        clean_text = raw_response.split("Confidence Score (0-1):")[-1]
        
        confidence = self._extract_confidence(clean_text)
        return float(1.0 - confidence)