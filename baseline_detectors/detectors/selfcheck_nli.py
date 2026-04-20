# baseline_detectors/detectors/selfcheck_nli.py
"""
SelfCheckNLI Detector - 灰盒 NLI 一致性检测器

原理：
    1. 将主回答 (Main Output) 拆分成单独的句子。
    2. 将每个句子作为 Hypothesis，多次采样的文本 (Stochastic Samples) 作为 Premise。
    3. 用 NLI 模型判断 Premise 是否能蕴含 (Entail) Hypothesis。
    4. 如果多次采样都无法蕴含该句子，说明该句子是幻觉。
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

# 关闭 httpx 刷屏
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

@register_detector("selfcheck_nli")
class SelfCheckNLIDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            nli_model: str = "roberta-large-mnli",  # 🎯 核心修复：换成绝对不会报错的 roberta
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # 声明依赖：需要多次采样文本
        self.requires_stochastic = True
        
        self.nli_model_name = nli_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.nli_pipeline = None

        logger.info(f"[{self.name}] SelfCheckNLI 初始化完成 (裁判模型: {self.nli_model_name})")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_nli_model(self):
        """安全加载 NLI 判定模型"""
        if self.nli_pipeline is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"[{self.name}] 正在加载 NLI 模型: {self.nli_model_name}...")
            
            # 🎯 拆分加载，断绝 pipeline 底层瞎搞的可能性
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
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
        """预热加载模型"""
        self._load_nli_model()

    def _split_into_sentences(self, text: str) -> List[str]:
        """简单的分句逻辑（如果是英文推荐用 spacy，这里用标点做 fallback）"""
        import re
        # 按句号、问号、感叹号及换行符分句
        sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.nli_pipeline is None:
            self._load_nli_model()

        # 1. 获取主回答和多次采样
        main_text = accessor.metadata.get("model_output_text", "")
        stochastic_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
        
        samples = []
        if isinstance(stochastic_data, dict):
            samples = stochastic_data.get("samples", [])
        elif isinstance(stochastic_data, list):
            samples = stochastic_data

        valid_samples = [s for s in samples if s and s.strip()]

        if not main_text or not valid_samples:
            return float('nan')

        try:
            # 2. 对主回答分句
            sentences = self._split_into_sentences(main_text)
            if not sentences:
                return float('nan')

            # 3. 构建 NLI 验证对：(Premise=采样文本, Hypothesis=主回答的一个句子)
            pairs = []
            pair_indices = [] # 记录 (sentence_idx, sample_idx)
            
            for s_idx, sentence in enumerate(sentences):
                for samp_idx, sample in enumerate(valid_samples):
                    pairs.append({"text": sample, "text_pair": sentence})
                    pair_indices.append((s_idx, samp_idx))

            # 4. 批量推理
            results = self.nli_pipeline(pairs)

            # 5. 解析分数：计算每个句子不被蕴含（Contradiction/Neutral）的概率
            # Roberta-large-mnli 的 label: 'CONTRADICTION', 'NEUTRAL', 'ENTAILMENT'
            sentence_hallucination_scores = np.zeros(len(sentences))
            
            for (s_idx, samp_idx), res in zip(pair_indices, results):
                label = res['label'].upper()
                score = res['score']
                
                # 如果是蕴含（Entailment），则幻觉风险为 0；否则将其计为有幻觉风险
                # 原论文做法：非 entail 的概率作为该次采样的幻觉得分
                is_entail = 'ENTAIL' in label or label == 'LABEL_2' # roberta LABEL_2 是 entailment
                
                if not is_entail:
                    # 如果不是 entail，加上权重（简单的做法是直接 +1，或者加 score）
                    sentence_hallucination_scores[s_idx] += 1.0

            # 6. 对每个句子取平均（跨多次采样的平均幻觉概率）
            num_samples = len(valid_samples)
            sentence_scores = sentence_hallucination_scores / num_samples

            # 7. 最终得分：整段话的最大句子幻觉得分 (SelfCheck 原论文推荐 max)
            final_score = np.max(sentence_scores)
            
            return float(final_score)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: SelfCheckNLI 计算失败: {e}")
            return float('nan')