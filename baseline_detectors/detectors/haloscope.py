"""
HaloScope Detector - 利用伪标签进行幻觉检测的强基线

核心工程对齐 (基于官方实现 hal_det_llama.py)：
    1. 提取生成回复的隐藏层特征 (Hidden States)。
    2. 中心化后进行 PCA / SVD 降维。
    3. 利用训练集(验证集)的真实标签，遍历搜索最优的特征维度 K 和投影方向 (Sign)。
    4. 计算每个样本投影向量的 L2 范数 (Magnitude)。
    5. 遍历百分位阈值，生成伪标签 (Pseudo-labels)，并找到使得 LR 验证表现最佳的阈值。
    6. 利用最佳伪标签在缩放后的特征上训练最终的分类器 (Logistic Regression)。
"""

import logging
import os
import sys
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logger = logging.getLogger(__name__)


@register_detector("haloscope")
class HaloScopeDetector(BaseDetector):
    def __init__(
        self,
        name: str,
        max_components: int = 15,
        target_layer: int = -1,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_components = max_components
        self.target_layer = target_layer

        self.pca = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

        self.is_fitted = False
        self.best_k = 1
        self.best_sign = 1
        self.feature_mean = None

        self.requires_qa_features = True

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        features_list = []
        true_labels = []

        # 1. 提取特征与真实标签
        for accessor in train_accessors:
            try:
                feature = accessor.get_hidden_states(
                    layer_idx=self.target_layer,
                    pooling="mean",
                )
                category = accessor.metadata.get("eval_category")
                if category in ["correct", "hallucination"] and feature is not None:
                    label = 1 if category == "hallucination" else 0
                    features_list.append(feature)
                    true_labels.append(label)
            except Exception:
                continue

        if not features_list:
            raise ValueError(f"[{self.name}] 没有提取到有效的特征！请确保 h5 文件完整。")

        X = np.array(features_list)
        y = np.array(true_labels)

        # 2. 特征中心化
        self.feature_mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - self.feature_mean

        # 3. PCA 降维
        max_k = min(self.max_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=max_k).fit(X_centered)
        X_pca = self.pca.transform(X_centered)

        # 阶段 A: 遍历寻找最优的投影维度 K 和投影方向 Sign
        best_svd_auroc = 0
        best_scores = None

        for k in range(1, max_k + 1):
            mags = np.linalg.norm(X_pca[:, :k], axis=1)
            try:
                auroc_pos = roc_auc_score(y, mags)
                auroc_neg = roc_auc_score(y, -mags)

                if auroc_pos > best_svd_auroc:
                    best_svd_auroc = auroc_pos
                    self.best_k = k
                    self.best_sign = 1
                    best_scores = mags

                if auroc_neg > best_svd_auroc:
                    best_svd_auroc = auroc_neg
                    self.best_k = k
                    self.best_sign = -1
                    best_scores = -mags
            except ValueError:
                pass

        if best_scores is None:
            self.best_k = max_k
            self.best_sign = 1
            best_scores = np.linalg.norm(X_pca[:, :max_k], axis=1)

        # 阶段 B: 遍历寻找最优的伪标签切分阈值 (Threshold)
        best_lr_auroc = 0
        best_pseudo_labels = None

        X_scaled = self.scaler.fit_transform(X)

        for pct in np.linspace(10, 90, 17):
            thres = np.percentile(best_scores, pct)
            pseudo_y = (best_scores > thres).astype(int)

            if len(set(pseudo_y)) < 2:
                continue

            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_scaled, pseudo_y)

            lr_preds = clf.predict_proba(X_scaled)[:, 1]
            try:
                lr_auroc = roc_auc_score(y, lr_preds)
                if lr_auroc > best_lr_auroc:
                    best_lr_auroc = lr_auroc
                    best_pseudo_labels = pseudo_y
            except ValueError:
                pass

        if best_pseudo_labels is None:
            best_pseudo_labels = y

        # 阶段 C: 最终模型定型
        self.classifier.fit(X_scaled, best_pseudo_labels)
        self.is_fitted = True

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] 模型尚未训练")

        try:
            feature = accessor.get_hidden_states(
                layer_idx=self.target_layer,
                pooling="mean",
            )
            if feature is None:
                return float('nan')

            feature_scaled = self.scaler.transform(feature.reshape(1, -1))
            prob_hallucination = self.classifier.predict_proba(feature_scaled)[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"[{self.name}] sample {accessor.sample_id} prediction failed: {e}")
            return float('nan')
