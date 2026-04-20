import numpy as np
from difflib import SequenceMatcher
from detectors.base import BaseDetector
from detectors.registry import register_detector

@register_detector("lexical_similarity")
class LexicalSimilarityDetector(BaseDetector):
    def __init__(self, name="lexical_similarity", num_samples=10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True # 需要多次采样文本
        self.num_samples = num_samples

    def _calculate_similarity(self, str1, str2):
        # 对齐源码常用的 SequenceMatcher (或可替换为 ROUGE-L)
        return SequenceMatcher(None, str1, str2).ratio()

    def predict_score(self, accessor):
        texts = accessor.get_stochastic_samples()
        if not texts or len(texts) < 2: return float('nan')
        
        texts = texts[:self.num_samples]
        sims = []
        # 对齐源码：两两对比 (Pairwise)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(self._calculate_similarity(texts[i], texts[j]))
        
        # 返回 1 - 平均相似度 (相似度越高，幻觉越低)
        avg_sim = np.mean(sims) if sims else 0.0
        return float(1.0 - avg_sim)