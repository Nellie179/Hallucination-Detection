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
        if not text: return "neutral"
        text = text.lower()
        neg_patterns = [r"final grade:\s*incorrect", r"is incorrect", r"\bincorrect\b", r"wrong", r"\bfalse\b",
                        r"\bno\b"]
        pos_patterns = [r"final grade:\s*correct", r"is correct", r"\bcorrect\b", r"accurate", r"\btrue\b", r"\byes\b"]

        for p in neg_patterns:
            if re.search(p, text): return "incorrect"
        for p in pos_patterns:
            if re.search(p, text): return "correct"
        return "neutral"

    def predict_score(self, accessor) -> float:
        logprobs = accessor.get_token_logprobs()
        if logprobs and len(logprobs) > 0:
            lp = float(logprobs[0])
            if not (math.isnan(lp) or math.isinf(lp)):
                p_true = np.exp(lp)
                p_true = min(max(p_true, 0.0), 1.0)
                return float(1.0 - p_true)

        raw_text = accessor.metadata.get("self_evaluator_raw", "")
        label = self._get_text_label(raw_text)

        if label == "correct":
            return 0.0
        elif label == "incorrect":
            return 1.0

        raise RuntimeError(
            f"Sample {accessor.sample_id}: No logprobs found in H5, and regex pattern matching failed for text ({raw_text})")