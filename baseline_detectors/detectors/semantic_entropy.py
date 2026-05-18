import numpy as np
import logging
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


@register_detector("semantic_entropy")
class SemanticEntropyDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            nli_model: str = "roberta-large-mnli",
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.requires_stochastic_logprobs = True

        self.nli_model_name = nli_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.nli_pipeline = None

        logger.info(f"[{self.name}] Semantic Entropy initialization complete (Referee model: {self.nli_model_name})")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_nli_model(self):
        if self.nli_pipeline is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"[{self.name}] Loading NLI model: {self.nli_model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            tokenizer.truncation_side = 'left'

            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name, use_safetensors=False)
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=16,
                truncation=True,
                max_length=512
            )
            logger.info(f"[{self.name}] ✓ NLI model loading complete")
        except ImportError:
            raise ImportError(f"[{self.name}] Missing transformers library")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        self._load_nli_model()

    def _build_equivalence_classes(self, samples: List[str]) -> List[List[int]]:
        n = len(samples)
        if n == 0: return []
        if n == 1: return [[0]]

        pairs = []
        pair_indices = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append({"text": samples[i], "text_pair": samples[j]})
                    pair_indices.append((i, j))

        results = self.nli_pipeline(pairs)

        E = np.zeros((n, n), dtype=bool)
        for (i, j), res in zip(pair_indices, results):
            label = res['label'].upper()
            if 'ENTAIL' in label or label == 'LABEL_2':
                E[i][j] = True

        unassigned = list(range(n))
        classes = []

        while unassigned:
            c = unassigned.pop(0)
            current_class = [c]
            to_remove = []

            for i in unassigned:
                if E[c][i] and E[i][c]:
                    current_class.append(i)
                    to_remove.append(i)

            for i in to_remove:
                unassigned.remove(i)

            classes.append(current_class)

        return classes

    def _get_samples_and_weights(self, accessor: SampleAccessor):
        import math

        raw_samples = accessor.metadata.get("stochastic_samples", [])
        raw_lls = accessor.metadata.get("stochastic_log_likelihoods", [])

        if not raw_samples and hasattr(accessor, "get_stochastic_samples"):
            raw_samples = accessor.get_stochastic_samples()
            raw_lls = accessor.get_stochastic_logprobs()

        valid_samples = []
        valid_lls = []
        has_likelihoods = bool(raw_lls)

        for i, s in enumerate(raw_samples):
            if s and str(s).strip():
                valid_samples.append(str(s))
                if has_likelihoods and i < len(raw_lls):
                    lp = raw_lls[i]
                    if lp is None or math.isnan(lp) or math.isinf(lp):
                        has_likelihoods = False
                    else:
                        valid_lls.append(float(lp))

        if not has_likelihoods or len(valid_lls) != len(valid_samples):
            has_likelihoods = False
            weights = np.ones(len(valid_samples)) / len(valid_samples)
        else:
            lls = np.array(valid_lls)
            lls_shifted = lls - np.max(lls)
            exp_lls = np.exp(lls_shifted)
            weights = exp_lls / np.sum(exp_lls)

        return valid_samples, weights, has_likelihoods

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.nli_pipeline is None:
            self._load_nli_model()

        valid_samples, weights, has_likelihoods = self._get_samples_and_weights(accessor)

        if len(valid_samples) < 2:
            return float('nan')

        try:
            classes = self._build_equivalence_classes(valid_samples)

            probs = np.zeros(len(classes))
            for cluster_idx, class_indices in enumerate(classes):
                for idx in class_indices:
                    probs[cluster_idx] += weights[idx]

            semantic_entropy = -np.sum(probs * np.log(probs + 1e-10))

            return float(semantic_entropy)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id} - Semantic Entropy computation failed: {e}")
            return float('nan')