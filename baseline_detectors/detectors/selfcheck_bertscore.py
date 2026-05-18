import numpy as np
import logging
from typing import List
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_list1(lst, n):
    expanded = []
    for item in lst:
        expanded.extend([item] * n)
    return expanded

def expand_list2(lst, n):
    expanded = []
    for _ in range(n):
        expanded.extend(lst)
    return expanded


@register_detector("selfcheck_bertscore")
class SelfCheckBERTScoreDetector(BaseDetector):

    def __init__(
            self,
            name: str,
            bert_model: str = "roberta-large",
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.requires_stochastic = True

        self.bert_model = bert_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")

        self.scorer = None
        self.nlp = None

        logger.info(f"[{self.name}] SelfCheckGPT(BERTScore) initialization complete")
        logger.info(f"  - Dependency declaration: requires_stochastic = True")
        logger.info(f"  - BERT model: {bert_model}")
        logger.info(f"  - Device: {self.device}")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_dependencies(self):
        if self.scorer is not None and self.nlp is not None:
            return

        try:
            import spacy
            from bert_score import BERTScorer

            logger.info(f"[{self.name}] Loading spacy model (en_core_web_sm)...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found, attempting automatic download...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            logger.info(f"[{self.name}] Loading BERTScore model: {self.bert_model}...")

            self.scorer = BERTScorer(
                model_type=self.bert_model,
                lang="en",
                rescale_with_baseline=False,
                device=self.device
            )

            logger.info(f"[{self.name}] ✓ Dependencies successfully loaded")

        except ImportError:
            raise ImportError(
                f"[{self.name}] Missing dependencies: bert-score or spacy\n"
                f"Please install: pip install bert-score spacy"
            )
        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to load dependencies: {e}")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        self._load_dependencies()

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.scorer is None or self.nlp is None:
            self._load_dependencies()

        main_output = accessor.get_model_output_text()
        samples = accessor.get_stochastic_samples()

        if not main_output or not main_output.strip():
            logger.debug(f"Sample {accessor.sample_id}: Main output is empty")
            return float('nan')

        valid_samples = [s for s in samples if s and s.strip()]
        if not valid_samples:
            logger.warning(f"Sample {accessor.sample_id}: Missing valid stochastic samples")
            return float('nan')

        try:
            sentences = [sent.text.strip() for sent in self.nlp(main_output).sents]
            sentences = [sent for sent in sentences if len(sent) > 0]
            num_sentences = len(sentences)

            if num_sentences == 0:
                return float('nan')

            num_samples = len(valid_samples)
            bertscore_array = np.zeros((num_sentences, num_samples))

            for s in range(num_samples):
                sample_passage = valid_samples[s]
                sentences_sample = [sent.text.strip() for sent in self.nlp(sample_passage).sents]
                sentences_sample = [sent for sent in sentences_sample if len(sent) > 0]
                num_sentences_sample = len(sentences_sample)

                if num_sentences_sample == 0:
                    continue

                refs = expand_list1(sentences, num_sentences_sample)
                cands = expand_list2(sentences_sample, num_sentences)

                P, R, F1 = self.scorer.score(cands, refs)

                F1_arr = F1.reshape(num_sentences, num_sentences_sample)
                F1_arr_max_axis1 = F1_arr.max(axis=1).values.numpy()
                bertscore_array[:, s] = F1_arr_max_axis1

            bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
            one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent

            return float(np.mean(one_minus_bertscore_mean_per_sent))

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: BERTScore computation failed:\n{traceback.format_exc()}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        if self.scorer is None or self.nlp is None:
            self._load_dependencies()

        main_output = accessor.get_model_output_text()
        samples = accessor.get_stochastic_samples()

        valid_samples = [s for s in samples if s and s.strip()]

        if not valid_samples:
            return {"error": "No valid samples"}

        try:
            sentences = [sent.text.strip() for sent in self.nlp(main_output).sents]
            sentences = [sent for sent in sentences if len(sent) > 0]
            num_sentences = len(sentences)

            if num_sentences == 0:
                return {"error": "No valid sentences to analyze"}

            num_samples = len(valid_samples)
            bertscore_array = np.zeros((num_sentences, num_samples))

            for s in range(num_samples):
                sample_passage = valid_samples[s]
                sentences_sample = [sent.text.strip() for sent in self.nlp(sample_passage).sents]
                sentences_sample = [sent for sent in sentences_sample if len(sent) > 0]
                num_sentences_sample = len(sentences_sample)

                if num_sentences_sample == 0:
                    continue

                refs = expand_list1(sentences, num_sentences_sample)
                cands = expand_list2(sentences_sample, num_sentences)

                P, R, F1 = self.scorer.score(cands, refs)
                F1_arr = F1.reshape(num_sentences, num_sentences_sample)
                bertscore_array[:, s] = F1_arr.max(axis=1).values.numpy()

            bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
            overall_avg_f1 = float(np.mean(bertscore_mean_per_sent))

            return {
                "main_output": main_output,
                "num_samples": num_samples,
                "f1_scores_per_sentence": bertscore_mean_per_sent.tolist(),
                "avg_f1": overall_avg_f1,
                "std_f1": float(np.std(bertscore_mean_per_sent)),
                "min_f1": float(np.min(bertscore_mean_per_sent)),
                "max_f1": float(np.max(bertscore_mean_per_sent)),
                "hallucination_score": 1.0 - overall_avg_f1
            }
        except Exception as e:
            return {"error": f"Analyze failed: {e}"}


if __name__ == "__main__":
    print("=" * 70)
    print("SelfCheckGPT (BERTScore) Unit Test")
    print("=" * 70)

    class MockAccessor(SampleAccessor):
        def __init__(self, sample_id, main_output, samples):
            self.sample_id = sample_id
            self.metadata = {"model_output_text": main_output}
            self.h5_group = None
            self.stochastic_samples_dict = {sample_id: samples}

    print("\n[Test 1] High consistency response (Expected: low hallucination score)")
    accessor1 = MockAccessor(
        sample_id="test_001",
        main_output="Paris is the capital of France.",
        samples=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "Paris serves as the capital city of France.",
            "France's capital is Paris.",
        ]
    )

    print("\n[Test 2] Low consistency response (Expected: high hallucination score)")
    accessor2 = MockAccessor(
        sample_id="test_002",
        main_output="The capital of France is Lyon.",
        samples=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "I think it might be Marseille.",
            "Bordeaux could be the capital.",
        ]
    )

    try:
        detector = SelfCheckBERTScoreDetector(
            name="test_bertscore",
            bert_model="microsoft/deberta-v3-small"
        )

        assert detector.requires_stochastic == True

        detector.fit([accessor1, accessor2])

        score1 = detector.predict_score(accessor1)
        analysis1 = detector.analyze(accessor1)
        print(f"Hallucination score: {score1:.3f} | Average F1: {analysis1['avg_f1']:.3f}")

        score2 = detector.predict_score(accessor2)
        analysis2 = detector.analyze(accessor2)
        print(f"Hallucination score: {score2:.3f} | Average F1: {analysis2['avg_f1']:.3f}")

        print("\n✅ Test complete")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()