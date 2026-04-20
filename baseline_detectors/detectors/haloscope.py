# baseline_detectors/detectors/haloscope.py
"""
HaloScope Detector - 利用无标注数据的幻觉检测

原理：
    利用部署中自然产生的无标注LLM生成数据进行幻觉检测。
    通过membership estimation自动区分truthful和hallucinated样本。

方法：
    1. 提取内部状态(hidden states)
    2. 使用SVD进行membership estimation
       - 将数据投影到主成分
       - 通过分数阈值分离两组
    3. 训练二分类器

基于官方实现：
    https://github.com/deeplearning-wisc/haloscope

参考文献：
    Du et al. "HaloScope: Harnessing Unlabeled LLM Generations for
    Hallucination Detection"
    NeurIPS 2024
"""

import numpy as np
import logging
from typing import List
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_detector("haloscope")
class HaloScopeDetector(BaseDetector):
    """
    HaloScope检测器

    需要数据：
        - Hidden states
        - 可以是无标注数据
    """

    def __init__(
        self,
        name: str,
        n_components: int = 10,  # PCA主成分数量
        threshold_percentile: float = 0.5,  # membership分离阈值百分位
        target_layer: int = -1,  # 使用哪一层
        **kwargs
    ):
        """
        Args:
            n_components: PCA主成分数量
            threshold_percentile: 用于分离truthful/hallucinated的阈值百分位
            target_layer: 目标层索引
        """
        super().__init__(name, **kwargs)

        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.target_layer = target_layer

        # 组件
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.is_fitted = False

        logger.info(f"[{self.name}] HaloScope检测器初始化完成")
        logger.info(f"  PCA成分: {n_components}")
        logger.info(f"  阈值百分位: {threshold_percentile}")

    def _membership_estimation(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Membership estimation: 区分truthful vs hallucinated

        基于官方实现:
        1. 中心化特征
        2. PCA投影
        3. 计算投影分数的范数
        4. 使用阈值分离

        Args:
            features: shape [n_samples, feature_dim]

        Returns:
            伪标签 [n_samples] (0=truthful, 1=hallucinated)
        """
        # 中心化
        centered = features - np.mean(features, axis=0, keepdims=True)

        # PCA投影
        if not hasattr(self.pca, 'components_'):
            # 首次fit PCA
            self.pca.fit(centered)

        projected = self.pca.transform(centered)

        # 计算投影分数(使用前几个主成分)
        # 通常使用第一主成分或所有成分的范数
        scores = np.linalg.norm(projected, axis=1)

        # 根据阈值分离
        threshold = np.percentile(scores, self.threshold_percentile * 100)

        # 低分数 -> truthful (0), 高分数 -> hallucinated (1)
        pseudo_labels = (scores > threshold).astype(int)

        return pseudo_labels

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        训练HaloScope

        可以使用无标注或有标注数据
        """
        logger.info(f"[{self.name}] 开始训练HaloScope...")

        # 提取特征
        features_list = []
        true_labels = []
        has_labels = False

        for accessor in train_accessors:
            try:
                # 提取hidden state
                feature = accessor.get_hidden_states(
                    layer_idx=self.target_layer,
                    pooling="mean"
                )
                features_list.append(feature)

                # 尝试获取真实标签(如果有的话)
                category = accessor.metadata.get("eval_category")
                if category in ["correct", "hallucination"]:
                    label = 1 if category == "hallucination" else 0
                    true_labels.append(label)
                    has_labels = True

            except Exception as e:
                logger.warning(f"样本 {accessor.sample_id} 处理失败: {e}")
                continue

        if len(features_list) == 0:
            raise ValueError("没有有效的训练样本!")

        features = np.array(features_list)

        logger.info(f"[{self.name}] 训练样本数: {len(features)}")
        logger.info(f"  特征维度: {features.shape[1]}")

        # 如果有真实标签,直接使用
        if has_labels and len(true_labels) == len(features):
            logger.info("  使用真实标签训练")
            labels = np.array(true_labels)
        else:
            # 否则使用membership estimation
            logger.info("  使用membership estimation生成伪标签")
            labels = self._membership_estimation(features)

        logger.info(f"  Truthful: {np.sum(labels == 0)}")
        logger.info(f"  Hallucinated: {np.sum(labels == 1)}")

        # 训练分类器
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels)
        self.is_fitted = True

        logger.info(f"[{self.name}] HaloScope训练完成")

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        预测幻觉分数

        Returns:
            float: 幻觉概率 [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] 模型尚未训练")

        try:
            # 提取特征
            feature = accessor.get_hidden_states(
                layer_idx=self.target_layer,
                pooling="mean"
            )

            # 预测
            feature_scaled = self.scaler.transform(feature.reshape(1, -1))
            prob_hallucination = self.classifier.predict_proba(feature_scaled)[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"样本 {accessor.sample_id} 预测失败: {e}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """详细分析"""
        try:
            feature = accessor.get_hidden_states(
                layer_idx=self.target_layer,
                pooling="mean"
            )

            return {
                "feature_dim": len(feature),
                "n_components": self.n_components,
                "hallucination_score": self.predict_score(accessor) if self.is_fitted else None
            }
        except Exception as e:
            return {"error": str(e)}


# 测试代码省略,与其他检测器类似
