import numpy as np
import logging
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logger = logging.getLogger(__name__)


@register_detector("sep")
class SEPDetector(BaseDetector):
    def __init__(self, name: str, target_layer: int = -1, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_qa_features = True
        self.target_layer = target_layer
        self.probe = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_features(self, accessor: SampleAccessor):
        h5_file = getattr(accessor, "qa_h5_file", None)
        if h5_file is None:
            raise ValueError(f"Sample {accessor.sample_id} is missing QA feature file handle")

        base_key = f"{accessor.sample_id}_sep"
        if base_key not in h5_file:
            raise KeyError(f"Base key '{base_key}' not found in H5 file")

        grp = h5_file[base_key]["sep_points"]

        all_keys = list(grp.keys())
        layers = [int(k.split('_')[-1]) for k in all_keys if k.startswith("slt_layer_")]

        if not layers:
            raise ValueError(f"No layer data found under {base_key}/sep_points")

        if self.target_layer >= 0:
            l_idx = self.target_layer
        else:
            l_idx = sorted(layers)[-1]

        tbg_key = f"tbg_layer_{l_idx}"
        slt_key = f"slt_layer_{l_idx}"

        if tbg_key not in grp or slt_key not in grp:
            tbg_key = f"tbg_layer_{l_idx:02d}"
            slt_key = f"slt_layer_{l_idx:02d}"
            if tbg_key not in grp:
                raise KeyError(f"Features for layer {l_idx} (TBG/SLT) not found")

        tbg_feat = np.array(grp[tbg_key])
        slt_feat = np.array(grp[slt_key])

        return np.concatenate([tbg_feat, slt_feat], axis=-1)

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        X_list, y_list = [], []
        for acc in train_accessors:
            try:
                feat = self._extract_features(acc)
                cat = acc.metadata.get("eval_category")
                if cat not in ["correct", "hallucination"]: continue
                X_list.append(feat)
                y_list.append(1 if cat == "hallucination" else 0)
            except Exception as e:
                logger.debug(f"SEP sample {acc.sample_id} fitting failed: {e}")
                continue

        if len(X_list) < 5:
            logger.warning(f"[{self.name}] Only {len(X_list)} valid training samples found, training aborted.")
            return

        X = np.array(X_list)
        X_scaled = self.scaler.fit_transform(X)
        self.probe = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.probe.fit(X_scaled, np.array(y_list))
        self.is_fitted = True
        logger.info(f"[{self.name}] SEP probe training complete (Samples: {len(y_list)})")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            feat = self._extract_features(accessor).reshape(1, -1)
            feat_scaled = self.scaler.transform(feat)
            return float(self.probe.predict_proba(feat_scaled)[0][1])
        except Exception:
            return float('nan')