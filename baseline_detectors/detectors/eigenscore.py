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
# 1. 内部隐状态版 (完全对齐官方逻辑，默认取最后一层)
# ---------------------------------------------------------
@register_detector("eigenscore_internal")
class EigenScoreInternalDetector(BaseDetector):
    def __init__(self, name: str = "eigenscore_internal", layer_idx: int = -1, num_samples: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.requires_stochastic_hidden_states = True
        self.layer_idx = layer_idx  # 💡 默认为 -1，支持外部传参自定义
        self.num_samples = num_samples
        self.alpha = 1e-3  # 💡 官方正则化系数

    def predict_score(self, accessor: SampleAccessor) -> float:
        try:
            vectors = []
            st_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
            texts = st_data.get("samples", []) if isinstance(st_data, dict) else st_data
            actual_num_samples = len(texts) if texts else self.num_samples

            for i in range(actual_num_samples):
                # 直接通过索引拉取，若越界由 accessor 报错或返回 None
                hs_raw = accessor.get_stochastic_hidden_states(sample_idx=i, layer_idx=self.layer_idx)
                
                if hs_raw is None or len(hs_raw) == 0:
                    continue
                    
                # 🛡️ 战时装甲：强制 float64 运算，预防大模型隐状态导致的数值溢出
                hs = np.array(hs_raw, dtype=np.float64)
                v = np.mean(hs, axis=0) 
                vectors.append(v)

            if len(vectors) < 2: 
                return float('nan')
            
            X = np.stack(vectors, axis=0)
            
            # 💡 官方逻辑：协方差矩阵 + Tikhonov 正则化
            CovMatrix = np.cov(X) 
            CovMatrix = CovMatrix + self.alpha * np.eye(CovMatrix.shape[0])
            
            # 💡 官方逻辑：SVD 求奇异值
            u, s, vT = np.linalg.svd(CovMatrix)
            
            # 💡 官方逻辑：对数均值得分 (指标越大，幻觉越重)
            # 增加 maximum 保护，彻底杜绝 log(0) 的极端异常
            indicator = np.mean(np.log10(np.maximum(s, 1e-12)))
            return float(indicator)
            
        except Exception as e:
            logger.error(f"Sample {accessor.sample_id} - EigenScore Internal 崩溃详情: {e}")
            return float('nan')


# ---------------------------------------------------------
# 2. 外部语义版 (完全对齐官方 getEigenIndicatorOutput)
# ---------------------------------------------------------
@register_detector("eigenscore_semantic")
class EigenScoreSemanticDetector(BaseDetector):
    def __init__(self, name: str = "eigenscore_semantic", num_samples: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_stochastic = True
        self.num_samples = num_samples
        self.alpha = 1e-3 
        self._model = None
        self.model_path = './data/weights/nli-roberta-large' 

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_path)
        return self._model

    def predict_score(self, accessor: SampleAccessor) -> float:
        try:
            texts = accessor.get_stochastic_samples()
            if not texts or len(texts) < 2: return float('nan')
            texts = texts[:self.num_samples]

            embeddings = self.model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype(np.float64)
            
            # 语义空间同样计算协方差
            CovMatrix = np.cov(embeddings)
            CovMatrix = CovMatrix + self.alpha * np.eye(CovMatrix.shape[0])
            
            u, s, vT = np.linalg.svd(CovMatrix)
            indicator = np.mean(np.log10(np.maximum(s, 1e-12)))
            return float(indicator)
            
        except Exception as e:
            logger.error(f"Semantic EigenScore Error: {e}")
            return float('nan')