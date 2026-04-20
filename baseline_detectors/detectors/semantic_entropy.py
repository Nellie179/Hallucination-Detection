# baseline_detectors/detectors/semantic_entropy.py
"""
Semantic Entropy Detector - 语义熵检测器
"""

import numpy as np
import logging
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

# 关闭 httpx 的烦人刷屏
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

@register_detector("semantic_entropy")
class SemanticEntropyDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            nli_model: str = "roberta-large-mnli",
            entailment_threshold: float = 0.5,
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.requires_stochastic_logprobs = True 

        self.nli_model_name = nli_model
        self.entailment_threshold = entailment_threshold
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.nli_pipeline = None

        logger.info(f"[{self.name}] Semantic Entropy 初始化完成")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_nli_model(self):
        """加载 NLI 判定模型"""
        if self.nli_pipeline is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"[{self.name}] 正在加载 NLI 模型: {self.nli_model_name}...")
            
            # 🎯 终极防崩溃修复：强制拆分，自己加载纯正的慢速 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            
            # 👈 核心修复：将截断方向改为左侧，确保保留长文本尾部的最终结论（短文本不受任何影响）
            tokenizer.truncation_side = 'left'
            
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
            # 将安全的对象直接喂给 pipeline，并开启截断保护
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,  
                device=0 if self.device == "cuda" else -1,
                batch_size=16,
                truncation=True,    # 👈 开启截断
                max_length=512      # 👈 限制最大长度为 512
            )
            logger.info(f"[{self.name}] ✓ NLI 模型加载完成")
        except ImportError:
            raise ImportError(f"[{self.name}] 缺少 transformers 库")

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
            # 兼容 Roberta 的标签输出 (ENTAILMENT)
            if ('ENTAIL' in label or label == 'LABEL_2' or label == 'LABEL_0') and res['score'] >= self.entailment_threshold:
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
        stochastic_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
        
        raw_samples = []
        raw_lls = []
        has_likelihoods = False

        if isinstance(stochastic_data, dict):
            raw_samples = stochastic_data.get("samples", [])
            raw_lls = stochastic_data.get("log_likelihoods", [])
            if raw_lls and len(raw_lls) == len(raw_samples):
                has_likelihoods = True
        elif isinstance(stochastic_data, list):
            raw_samples = stochastic_data

        valid_samples = []
        valid_lls = []
        for i, s in enumerate(raw_samples):
            if s and s.strip():
                valid_samples.append(s)
                if has_likelihoods:
                    valid_lls.append(raw_lls[i])

        if not valid_lls or len(valid_lls) != len(valid_samples):
            has_likelihoods = False
            weights = np.ones(len(valid_samples)) / len(valid_samples)
        else:
            # 清理可能存在的 None 值
            clean_lls = []
            for l in valid_lls:
                clean_lls.append(float(l) if l is not None else -100.0)
                
            lls = np.array(clean_lls)
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
            max_entropy = np.log(len(valid_samples))
            normalized_entropy = semantic_entropy / max_entropy if max_entropy > 0 else 0.0

            return float(normalized_entropy)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: 语义熵计算失败: {e}")
            return float('nan')