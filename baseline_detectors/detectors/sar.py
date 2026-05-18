import os
import math
import torch
import numpy as np
import logging
from typing import List

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logger = logging.getLogger(__name__)

try:
    from sentence_transformers.cross_encoder import CrossEncoder

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers is not installed, SAR probe will not function.")


@register_detector("sar")
class SARDetector(BaseDetector):
    def __init__(
            self,
            name="sar",
            measurement_model: str = "cross-encoder/stsb-distilroberta-base",
            t: float = 0.001,
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.t = t

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SAR strictly depends on sentence_transformers. Please execute: pip install sentence_transformers")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[{self.name}] Loading NLI model: {measurement_model} to {self.device}")
        self.measure_model = CrossEncoder(model_name=measurement_model, device=self.device, num_labels=1)

    def _semantic_weighted_log(self, similarities: List[List[float]], entropies: torch.Tensor) -> torch.Tensor:
        log_probs = -1 * entropies

        max_log_prob = log_probs.max()
        if torch.isinf(max_log_prob):
            return torch.zeros_like(entropies)

        shifted_log_probs = log_probs - max_log_prob
        shifted_probs = torch.exp(shifted_log_probs)

        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(shifted_probs, entropies)):
            sim_tensor = torch.tensor(similarities[idx], device=self.device)
            sim_tensor = torch.nan_to_num(sim_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

            other_probs = torch.cat([shifted_probs[:idx], shifted_probs[idx + 1:]])

            sum_term = prob + ((sim_tensor / self.t) * other_probs).sum()

            sum_term = torch.clamp(sum_term, min=1e-10)

            w_ent = -(torch.log(sum_term) + max_log_prob)
            weighted_entropy.append(w_ent)

        return torch.tensor(weighted_entropy, device=self.device)

    def predict_score(self, accessor: SampleAccessor) -> float:
        prompt = accessor.get_prompt_text()
        raw_samples = accessor.get_stochastic_samples()
        raw_logprobs = accessor.get_stochastic_logprobs()

        if not raw_samples or not raw_logprobs:
            logger.error(
                f"[SAR Critical Intercept] Sample {accessor.sample_id} does not contain any sampled data or probabilities!")
            logger.error(f"  - raw_samples length: {len(raw_samples) if raw_samples else 'None'}")
            logger.error(f"  - raw_logprobs length: {len(raw_logprobs) if raw_logprobs else 'None'}")
            return 0.5

        samples, logprobs = [], []
        for s, lp in zip(raw_samples, raw_logprobs):
            if lp is not None and not math.isnan(lp) and not math.isinf(lp):
                samples.append(s)
                logprobs.append(float(lp))

        num_generations = len(samples)
        if num_generations <= 1:
            logger.error(
                f"[SAR Critical Intercept] Sample {accessor.sample_id} has fewer than 2 valid generations and cannot be evaluated.")
            return 0.5

        print(
            f"[*] Sample {accessor.sample_id} data verification passed (Generations: {num_generations}), preparing tensor computations...")

        gen_entropies = torch.tensor([-lp for lp in logprobs], dtype=torch.float32, device=self.device)

        pairs = []
        pair_indices = []
        for i in range(num_generations):
            for j in range(i + 1, num_generations):
                pairs.append([prompt + samples[i], prompt + samples[j]])
                pair_indices.append((i, j))

        flat_similarities = self.measure_model.predict(pairs, show_progress_bar=False)

        similarities = {i: [] for i in range(num_generations)}
        for (i, j), sim_score in zip(pair_indices, flat_similarities):
            similarities[i].append(float(sim_score))
            similarities[j].append(float(sim_score))

        sar_scores = self._semantic_weighted_log(similarities, gen_entropies)
        final_score = float(sar_scores.mean().cpu().item())

        return final_score