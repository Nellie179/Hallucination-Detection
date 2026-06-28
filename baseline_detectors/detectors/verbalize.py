import re
import logging
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector("verbalize")
class VerbalizeDetector(BaseDetector):
    def __init__(self, name="verbalize", **kwargs):
        super().__init__(name, **kwargs)
        # Actual prompt is defined in generate_auxiliary_evals.py

    def _extract_confidence(self, text: str) -> float:
        if not text: return 0.5
        text = str(text).lower()

        match = re.search(r'(?:confidence|score|certainty).*?([0-9]*\.?[0-9]+%?)', text)

        if match:
            num_str = match.group(1)
        else:
            matches = re.findall(r"\b(\d+%\.?\d*|0?\.\d+|1\.0|1|0)\b", text)
            if not matches: return 0.5
            num_str = matches[-1]

        try:
            if "%" in num_str:
                val = float(num_str.replace("%", "")) / 100.0
            else:
                val = float(num_str)
                if val > 1.0 and val <= 10.0:
                    val = val / 10.0
                elif val > 10.0 and val <= 100.0:
                    val = val / 100.0

            return min(max(val, 0.0), 1.0)
        except Exception:
            return 0.5

    def predict_score(self, accessor) -> float:
        raw_response = accessor.metadata.get("verbalize_response", None)

        if not raw_response:
            return float('nan')

        clean_text = raw_response.split("Confidence Score (0-1):")[-1]

        confidence = self._extract_confidence(clean_text)
        return float(1.0 - confidence)