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
        强化版正则提取逻辑：
        寻找类似 0.8, .8, 80%, 1, 0 或 'Confidence: 0.9' 中的数字。
        解决了原版无法识别单独的 \"1\" 或 \"0\" 的致命 Bug。
        """
        if text is None: return 0.0
        
        # 匹配规则优先级：百分比 > 浮点数 > 单独的 1 或 0 (防止边界值漏抓)
        matches = re.findall(r"\d+%\.?\d*|0?\.\d+|1\.0|1|0", str(text))
        if not matches:
            return 0.5 # 源码兜底逻辑
        
        last_match = matches[-1] # 通常取最后一个出现的数字
        try:
            if "%" in last_match:
                return float(last_match.replace("%", "")) / 100.0
            return float(last_match)
        except ValueError:
            return 0.5

    def predict_score(self, accessor) -> float:
        """
        从 metadata 中读取预先生成的 verbalize_response 并解析
        """
        # 1. 尝试从 metadata 中获取模型当时的“自评回复”
        raw_response = accessor.metadata.get("verbalize_response", None)
        
        # 2. 如果不存在，说明 runner 生成阶段漏掉了这个字段
        if raw_response is None:
            logger.warning(f"Sample {accessor.sample_id}: Missing verbalize_response in metadata.")
            return float('nan')
        
        # 3. 解析置信度
        confidence = self._extract_confidence(raw_response)
        
        # 4. 转换：1.0 - 置信度 = 幻觉分 (对齐 Benchmark 指标)
        return float(1.0 - confidence)