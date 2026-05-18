import numpy as np
from rouge_score import rouge_scorer
from detectors.base import BaseDetector
from detectors.registry import register_detector


@register_detector("lexical_similarity")
class LexicalSimilarityDetector(BaseDetector):
    def __init__(self, name="lexical_similarity", num_samples=10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.num_samples = num_samples
        self.rougeEvaluator = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _calculate_similarity(self, str1, str2):
        results = self.rougeEvaluator.score(target=str1, prediction=str2)
        return results["rougeL"].fmeasure

    def predict_score(self, accessor):
        texts = accessor.get_stochastic_samples()
        if not texts or len(texts) < 2: return float('nan')

        texts = texts[:self.num_samples]
        sims = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(self._calculate_similarity(texts[i], texts[j]))

        avg_sim = np.mean(sims) if sims else 0.0
        return float(1.0 - avg_sim)