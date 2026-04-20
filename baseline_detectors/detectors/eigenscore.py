# baseline_detectors/detectors/eigenscore.py
import numpy as np
import torch
import logging
from typing import List, Any
from sentence_transformers import SentenceTransformer
from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 内部隐状态版 (完全对齐源码 eigenIndicator)
# ---------------------------------------------------------
@register_detector("eigenscore_internal")
class EigenScoreInternalDetector(BaseDetector):
    def __init__(self, name: str = "eigenscore_internal", layer_idx: int = -1, num_samples: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.requires_stochastic_hidden_states = True
        self.layer_idx = layer_idx
        self.num_samples = num_samples

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        对齐源码 getEigenIndicator_v0 逻辑:
        1. 提取 N 次采样的 Hidden States
        2. 对每个采样序列做 Mean Pooling (源码常用 get_num_tokens 后的均值)
        3. 计算特征值
        """
        try:
            vectors = []
            
            # 🛠️ [修复 1]: 动态探测真实的采样数量，放弃写死的 self.num_samples
            st_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
            texts = st_data.get("samples", []) if isinstance(st_data, dict) else st_data
            actual_num_samples = len(texts) if texts else self.num_samples

            for i in range(actual_num_samples):
                hs = accessor.get_stochastic_hidden_states(sample_idx=i, layer_idx=self.layer_idx)
                
                # 🛠️ [修复 2]: 防御性拦截，防止拿到 None 或空数组后报错
                if hs is None or len(hs) == 0:
                    continue
                    
                v = np.mean(hs, axis=0) 
                vectors.append(v)

            if len(vectors) < 2: 
                print(f"[-] 样本 {accessor.sample_id} 的有效隐状态数量不足 2 个，无法计算协方差")
                return float('nan')
            
            X = np.stack(vectors, axis=0) # [N, D]
            
            C = np.cov(X) # 默认 rowvar=True，得出 [N, N] 的样本间协方差
            e, _ = np.linalg.eigh(C) 
            
            # 🛠️ [修复 3]: 防御特征值全为 0 甚至负数导致的计算异常
            e_sum = np.sum(e)
            if e_sum <= 0:
                return 0.0 # 若无方差，视作极度确信，无幻觉风险

            score = 1.0 - (np.max(e) / (e_sum + 1e-10))
            return float(score)
            
        except Exception as e:
            # 🛠️ [修复 4]: 废除静默失败，强制暴露真凶
            logger.error(f"Sample {accessor.sample_id} - EigenScore Internal 崩溃详情: {e}")
            return float('nan')
# ---------------------------------------------------------
# 2. 外部语义版 (完全对齐源码 eigenIndicatorOutput)
# ---------------------------------------------------------
@register_detector("eigenscore_semantic")
class EigenScoreSemanticDetector(BaseDetector):
    def __init__(self, name: str = "eigenscore_semantic", num_samples: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.num_samples = num_samples
        self._model = None
        # 🎯 强制对齐源码使用的模型路径
        self.model_path = './data/weights/nli-roberta-large' 

    @property
    def model(self):
        if self._model is None:
            # 如果本地没路径，会自动从 HF 下载，但路径名保持一致
            self._model = SentenceTransformer(self.model_path)
        return self._model

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        对齐源码 getEigenIndicatorOutput 逻辑
        """
        try:
            texts = accessor.get_stochastic_samples()
            if not texts or len(texts) < 2: return float('nan')
            texts = texts[:self.num_samples]

            # 🎯 源码逻辑：直接使用 SentenceTransformer 编码文本
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # 计算相似度矩阵的特征值
            # 源码通常不手动做 Norm，而是直接 dot 之后看分布
            C = np.dot(embeddings, embeddings.T)
            e = np.linalg.eigvalsh(C)
            
            indicator = 1.0 - (e[-1] / (np.sum(e) + 1e-10))
            return float(indicator)
        except Exception as e:
            logger.error(f"Semantic EigenScore Error: {e}")
            return float('nan')