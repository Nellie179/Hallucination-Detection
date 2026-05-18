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


@register_detector("selfcheck_nli")
class SelfCheckNLIDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            nli_model: str = "roberta-large-mnli",
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.requires_stochastic = True

        self.nli_model_name = nli_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.nli_pipeline = None
        self.nlp = None

        logger.info(f"[{self.name}] SelfCheckNLI initialization complete (Referee model: {self.nli_model_name})")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_dependencies(self):
        if self.nli_pipeline is not None and self.nlp is not None:
            return
        try:
            import spacy
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found, attempting automatic download...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            logger.info(f"[{self.name}] Loading NLI model: {self.nli_model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)

            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=16,
                truncation=True,
                max_length=512,
                top_k=None
            )
            logger.info(f"[{self.name}] ✓ NLI model loading complete")
        except ImportError as e:
            raise ImportError(f"[{self.name}] Missing dependency library: {e}")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        self._load_dependencies()

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.nli_pipeline is None or self.nlp is None:
            self._load_dependencies()

        main_text = accessor.metadata.get("model_output_text", "").strip()
        stochastic_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
        samples = stochastic_data.get("samples", []) if isinstance(stochastic_data, dict) else stochastic_data
        valid_samples = [s for s in samples if s and s.strip()]

        if not main_text or not valid_samples:
            return float('nan')

        choices = accessor.metadata.get("structured_data", {}).get("choices", {})

        def smart_expand(text):
            clean_text = text.strip().upper()
            if len(clean_text) <= 2 and clean_text in choices:
                return f"The answer is {choices[clean_text]}"
            return text

        try:
            expanded_main = smart_expand(main_text)

            sentences = [sent.text.strip() for sent in self.nlp(expanded_main).sents]
            sentences = [sent for sent in sentences if len(sent) > 0]
            if not sentences:
                return float('nan')

            pairs = []
            pair_indices = []

            for s_idx, sentence in enumerate(sentences):
                for samp_idx, sample in enumerate(valid_samples):
                    expanded_sample = smart_expand(sample)
                    pairs.append({"text": expanded_sample, "text_pair": sentence})
                    pair_indices.append((s_idx, samp_idx))

            results = self.nli_pipeline(pairs)

            sentence_hallucination_scores = np.zeros(len(sentences))

            for (s_idx, samp_idx), res_list in zip(pair_indices, results):
                contradiction_prob = 0.0

                for class_score in res_list:
                    label = class_score['label'].upper()
                    if 'CONTRADICTION' in label or label == 'LABEL_0':
                        contradiction_prob = class_score['score']
                        break

                sentence_hallucination_scores[s_idx] += contradiction_prob

            num_samples = len(valid_samples)
            sentence_scores = sentence_hallucination_scores / num_samples

            final_score = np.max(sentence_scores)

            return float(final_score)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: SelfCheckNLI computation failed: {e}")
            return float('nan')