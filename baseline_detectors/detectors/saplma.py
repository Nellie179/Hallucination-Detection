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
            target_layer: int = -1,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("SAPLMA requires scikit-learn support")

        self.requires_qa_features = True
        self.target_layer = target_layer

        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        self.is_fitted = False

        logger.info(f"[{self.name}] SAPLMA (MLP) detector initialization complete")

    def _extract_hidden_state(self, accessor: SampleAccessor) -> np.ndarray:
        if not accessor.qa_h5_file:
            raise ValueError(f"Sample {accessor.sample_id} is missing QA features!")

        grp_name = f"{accessor.sample_id}_base_logit_recovery"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"base_logit_recovery group for sample {accessor.sample_id} is missing in H5 file.")

        grp = accessor.qa_h5_file[grp_name]

        layer_str = f"layer_{self.target_layer}"
        if self.target_layer < 0:
            layers = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("layer_")]
            layer_str = f"layer_{max(layers)}"

        feat = np.array(grp[layer_str], dtype=np.float32)
        return feat

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] Starting SAPLMA MLP network training...")
        X_list, y_list = [], []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]: continue

                feat = self._extract_hidden_state(accessor)
                X_list.append(feat)
                y_list.append(1 if category == "hallucination" else 0)
            except Exception as e:
                logger.debug(f"Skipping sample {accessor.sample_id}: {e}")
                continue

        if len(set(y_list)) < 2:
            logger.warning(
                f"[{self.name}] Training set must contain both positive and negative samples, skipping training.")
            return

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)
        self.is_fitted = True

        logger.info(f"[{self.name}] Training complete! (Fitting accuracy: {self.mlp.score(X_scaled, y) * 100:.2f}%)")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            feat = self._extract_hidden_state(accessor)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            return float(self.mlp.predict_proba(feat_scaled)[0][1])
        except Exception as e:
            return float('nan')