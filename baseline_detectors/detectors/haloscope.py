import numpy as np
import logging
from typing import List
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
    def __init__(
            self,
            name: str,
            max_components: int = 15,
            target_layer: int = -1,
            **kwargs
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
        logger.info(f"[{self.name}] Starting HaloScope training...")

        features_list = []
        true_labels = []

        for accessor in train_accessors:
            try:
                feature = accessor.get_hidden_states(
                    layer_idx=self.target_layer,
                    pooling="mean"
                )
                category = accessor.metadata.get("eval_category")
                if category in ["correct", "hallucination"] and feature is not None:
                    label = 1 if category == "hallucination" else 0
                    features_list.append(feature)
                    true_labels.append(label)
            except Exception:
                continue

        if not features_list:
            raise ValueError(f"[{self.name}] No valid features extracted! Please ensure the h5 file is complete.")

        X = np.array(features_list)
        y = np.array(true_labels)

        logger.info(f"[{self.name}] Training set loaded | Shape: {X.shape} | Hallucination samples: {sum(y)}")

        self.feature_mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - self.feature_mean

        max_k = min(self.max_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=max_k).fit(X_centered)
        X_pca = self.pca.transform(X_centered)

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

        logger.info(
            f"[{self.name}] Parameter search complete -> Best K: {self.best_k}, Sign: {self.best_sign}, SVD AUROC: {best_svd_auroc:.4f}")

        best_lr_auroc = 0
        best_pseudo_labels = None
        best_threshold_pct = 0.5

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
                    best_threshold_pct = pct / 100.0
            except ValueError:
                pass

        if best_pseudo_labels is None:
            best_pseudo_labels = y

        logger.info(
            f"[{self.name}] Pseudo-label assignment complete -> Best split point: Top {best_threshold_pct * 100:.1f}%, Validation AUROC: {best_lr_auroc:.4f}")

        self.classifier.fit(X_scaled, best_pseudo_labels)
        self.is_fitted = True

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] Model has not been trained yet")

        try:
            feature = accessor.get_hidden_states(
                layer_idx=self.target_layer,
                pooling="mean"
            )
            if feature is None:
                return float('nan')

            feature_scaled = self.scaler.transform(feature.reshape(1, -1))
            prob_hallucination = self.classifier.predict_proba(feature_scaled)[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"[{self.name}] Sample {accessor.sample_id} prediction failed: {e}")
            return float('nan')