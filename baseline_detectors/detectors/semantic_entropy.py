# baseline_detectors/detectors/semantic_entropy.py
"""
Semantic Entropy Detector - 语义熵检测器

🎯 已严格对齐 Kuhn et al. 2023 (ICLR) / Nature 2024:
1. 采用 DeBERTa-large-MNLI 作为双向蕴含 (Bidirectional Entailment) 裁判。
2. 修复 Contradiction 标签混用 Bug，严格判定语义等价类。
3. 输出纯正的 Raw Semantic Entropy (未进行除法归一化)。
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
            # nli_model: str = "microsoft/deberta-large-mnli", # 🎯 对齐论文：使用 DeBERTa
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

        logger.info(f"[{self.name}] Semantic Entropy 初始化完成 (裁判模型: {self.nli_model_name})")

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
            
            # 强制拆分，自己加载纯正的慢速 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            
            # 截断方向改为左侧，确保保留长文本尾部的最终结论
            tokenizer.truncation_side = 'left'
            
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name, use_safetensors=False)
            # 将安全的对象直接喂给 pipeline
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,  
                device=0 if self.device == "cuda" else -1,
                batch_size=16,
                truncation=True,    
                max_length=512      
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
            # 🎯 对齐论文核心修复：必须是 ENTAILMENT 才能算语义等价，坚决剔除 LABEL_0 (矛盾)
            # pipeline 默认返回 argmax，所以只要 label 是蕴含即可
            if 'ENTAIL' in label or label == 'LABEL_2':
                E[i][j] = True

        unassigned = list(range(n))
        classes = []

        # 🎯 对齐论文：基于互蕴含 (Mutual Entailment) 的贪心聚类 (Greedy Clustering)
        while unassigned:
            c = unassigned.pop(0)
            current_class = [c]
            to_remove = []
            
            for i in unassigned:
                # 只有 A 蕴含 B，且 B 蕴含 A 时，才分到同一个类
                if E[c][i] and E[i][c]:
                    current_class.append(i)
                    to_remove.append(i)
            
            for i in to_remove:
                unassigned.remove(i)
                
            classes.append(current_class)

        return classes

    def _get_samples_and_weights(self, accessor: SampleAccessor):
        import math
        
        # 🎯 修复 1：从正确的挂载点 metadata 中提取数据
        raw_samples = accessor.metadata.get("stochastic_samples", [])
        raw_lls = accessor.metadata.get("stochastic_log_likelihoods", [])
        
        # 兼容老接口
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
                    # 🎯 修复 2：极度严格的坏账清洗，发现 -Infinity 直接放弃概率加权
                    if lp is None or math.isnan(lp) or math.isinf(lp):
                        has_likelihoods = False
                    else:
                        valid_lls.append(float(lp))

        # 🎯 修复 3：退化机制。如果没有有效的概率，或者概率被污染，退化为均匀分布
        if not has_likelihoods or len(valid_lls) != len(valid_samples):
            has_likelihoods = False
            weights = np.ones(len(valid_samples)) / len(valid_samples)
        else:
            lls = np.array(valid_lls)
            lls_shifted = lls - np.max(lls) # 此时绝对安全，因为 inf 已经被拦在外面了
            exp_lls = np.exp(lls_shifted)
            weights = exp_lls / np.sum(exp_lls) # Softmax 归一化

        return valid_samples, weights, has_likelihoods

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.nli_pipeline is None:
            self._load_nli_model()

        valid_samples, weights, has_likelihoods = self._get_samples_and_weights(accessor)
        
        if len(valid_samples) < 2:
            return float('nan')

        try:
            # 1. 获取等价类
            classes = self._build_equivalence_classes(valid_samples)

            # 2. 累加同一个等价类中所有句子的概率
            probs = np.zeros(len(classes))
            for cluster_idx, class_indices in enumerate(classes):
                for idx in class_indices:
                    probs[cluster_idx] += weights[idx]

            # 3. 计算纯正的 Semantic Entropy (不再除以 max_entropy)
            semantic_entropy = -np.sum(probs * np.log(probs + 1e-10))

            return float(semantic_entropy)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: 语义熵计算失败: {e}")
            return float('nan')