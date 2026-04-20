# baseline_detectors/detectors/saplma.py
"""
SAPLMA (The Internal State of an LLM Knows When It's Lying)
利用多层感知机 (MLP) 对大模型回答时的隐藏层特征进行二分类有监督训练。
"""

import numpy as np
import logging
from typing import List
import sys
import os

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logger = logging.getLogger(__name__)

@register_detector("saplma")
class SAPLMADetector(BaseDetector):
    def __init__(
        self,
        name: str = "saplma",
        target_layer: int = -1,  # 默认取最后一层
        **kwargs
    ):
        super().__init__(name, **kwargs)
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("SAPLMA 需要 scikit-learn 支持")

        self.requires_qa_features = True
        self.target_layer = target_layer
        
        # 🎯 SAPLMA 核心：多层感知机 (隐藏层维度 256 -> 128)
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1000,
            early_stopping=True,  # 防止过拟合
            random_state=42
        )
        self.is_fitted = False

        logger.info(f"[{self.name}] SAPLMA (MLP) 探测器初始化完成")

    def _extract_hidden_state(self, accessor: SampleAccessor) -> np.ndarray:
        """读取大管家挂载过来的 base_logit_recovery 文件中的特征"""
        if not accessor.qa_h5_file:
            raise ValueError(f"样本 {accessor.sample_id} 缺少 QA 特征！")
            
        # 💡 极其聪明的白嫖：寻找 base_logit_recovery 的组
        grp_name = f"{accessor.sample_id}_base_logit_recovery"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"H5 中缺失样本 {accessor.sample_id} 的 base_logit_recovery 组，白嫖失败！")
            
        grp = accessor.qa_h5_file[grp_name]
        
        layer_str = f"layer_{self.target_layer}"
        if self.target_layer < 0:
            layers = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("layer_")]
            layer_str = f"layer_{max(layers)}"

        feat = np.array(grp[layer_str], dtype=np.float32)
        return feat

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] 开始训练 SAPLMA MLP网络...")
        X_list, y_list = [], []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]: continue

                feat = self._extract_hidden_state(accessor)
                X_list.append(feat)
                y_list.append(1 if category == "hallucination" else 0)
            except Exception as e:
                logger.debug(f"跳过样本 {accessor.sample_id}: {e}")
                continue

        if len(set(y_list)) < 2:
            logger.warning(f"[{self.name}] 训练集必须包含正负样本，跳过训练。")
            return

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"[{self.name}] 训练完成！(拟合准确率: {self.mlp.score(X_scaled, y)*100:.2f}%)")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            feat = self._extract_hidden_state(accessor)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            return float(self.mlp.predict_proba(feat_scaled)[0][1])
        except Exception as e:
            return float('nan')