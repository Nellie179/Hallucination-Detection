import json
import logging
import os
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector("tsv")
class TSVDetector(BaseDetector):
    def __init__(self, name="tsv", **kwargs):
        super().__init__(name, **kwargs)
        self.tsv_scores_cache = {}
        self.cache_loaded = False

    def _load_scores_if_needed(self, accessor):
        if self.cache_loaded: return

        base_dir = "."
        if hasattr(accessor, 'h5_group') and accessor.h5_group is not None:
            base_dir = os.path.dirname(accessor.h5_group.file.filename)
        elif hasattr(accessor, 'stochastic_h5_group') and accessor.stochastic_h5_group is not None:
            base_dir = os.path.dirname(accessor.stochastic_h5_group.file.filename)

        tsv_file = os.path.join(base_dir, "05_qa_features_tsv.jsonl")

        if os.path.exists(tsv_file):
            with open(tsv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.tsv_scores_cache[str(item["sample_id"])] = item["tsv_hallucination_score"]
            logger.info(f"[TSV] Successfully loaded {len(self.tsv_scores_cache)} scores from {base_dir}")
        else:
            logger.warning(
                f"[TSV] Feature file not found at: {tsv_file}. Verify that feature extraction has been executed.")

        self.cache_loaded = True

    def predict_score(self, accessor) -> float:
        self._load_scores_if_needed(accessor)
        sid = str(accessor.sample_id)
        return float(self.tsv_scores_cache.get(sid, float('nan')))