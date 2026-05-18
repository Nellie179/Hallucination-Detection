import numpy as np
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector("perplexity")
class PerplexityDetector(BaseDetector):
    def __init__(self, name="perplexity", **kwargs):
        super().__init__(name, **kwargs)
        self.requires_logprobs = True

    def predict_score(self, accessor) -> float:
        try:
            logprobs = getattr(accessor, "recovered_logprobs", None)
            if logprobs is None:
                logprobs = accessor.get_token_logprobs()

            if logprobs is None or len(logprobs) == 0:
                return float('nan')

            valid_logprobs = [float(p) for p in logprobs if p is not None and not np.isnan(float(p))]
            if not valid_logprobs:
                return float('nan')

            neg_log_likelihood = -np.mean(valid_logprobs)

            if neg_log_likelihood > 50:
                return float(1e10)

            return float(np.exp(neg_log_likelihood))
        except Exception as e:
            logger.debug(f"[{self.name}] Sample {accessor.sample_id} PPL computation failed: {e}")
            return float('nan')


@register_detector("ln_entropy")
class LNEntropyDetector(BaseDetector):
    def __init__(self, name="ln_entropy", **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.requires_logprobs = True

    def predict_score(self, accessor) -> float:
        try:
            st_logprobs = accessor.get_stochastic_logprobs()

            if st_logprobs and len(st_logprobs) > 0:
                valid_st_lps = [float(p) for p in st_logprobs if p is not None and not np.isnan(float(p))]

                if len(valid_st_lps) > 0:
                    expected_ln_entropy = -np.mean(valid_st_lps)
                    return float(expected_ln_entropy)

            base_lp = getattr(accessor, "recovered_logprobs", None)
            if base_lp is None:
                base_lp = accessor.get_token_logprobs()

            if base_lp is not None and len(base_lp) > 0:
                valid_base_lps = [float(p) for p in base_lp if p is not None and not np.isnan(float(p))]
                if len(valid_base_lps) > 0:
                    return float(-np.mean(valid_base_lps))

            return float('nan')

        except Exception as e:
            logger.debug(f"[{self.name}] Sample {accessor.sample_id} Entropy computation failed: {e}")
            return float('nan')