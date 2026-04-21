# baseline_detectors/detectors/sep.py
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
            raise ValueError(f"样本 {accessor.sample_id} 缺失 QA 特征文件句柄")

        # 🎯 核心修复 1: 路径对齐
        # 提取端存的是 {sid}_sep/sep_points/
        base_key = f"{accessor.sample_id}_sep"
        if base_key not in h5_file:
            raise KeyError(f"H5 中找不到基础 Key: {base_key}")
            
        grp = h5_file[base_key]["sep_points"] # 👈 深入到子组
        
        # 🎯 核心修复 2: 查找现有的层级
        # 提取端存的是 tbg_layer_X 和 slt_layer_X
        all_keys = list(grp.keys())
        layers = [int(k.split('_')[-1]) for k in all_keys if k.startswith("slt_layer_")]
        
        if not layers:
            raise ValueError(f"在 {base_key}/sep_points 下未找到任何层级数据")

        # 🎯 鲁棒性排序：取数字最大的那一层（不管提取了哪几层）
        if self.target_layer >= 0:
            l_idx = self.target_layer
        else:
            l_idx = sorted(layers)[-1]
            
        tbg_key = f"tbg_layer_{l_idx}"
        slt_key = f"slt_layer_{l_idx}"
        
        if tbg_key not in grp or slt_key not in grp:
            # 兼容带补零的格式 (layer_02)
            tbg_key = f"tbg_layer_{l_idx:02d}"
            slt_key = f"slt_layer_{l_idx:02d}"
            if tbg_key not in grp:
                raise KeyError(f"找不到层级 {l_idx} 的特征 (TBG/SLT)")

        tbg_feat = np.array(grp[tbg_key])
        slt_feat = np.array(grp[slt_key])

        # 🎯 算法对齐：SEP 的标准做法是拼接 TBG 和 SLT 特征
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
                # 显式打印报错，不再静默
                logger.debug(f"SEP 样本 {acc.sample_id} 拟合失败: {e}")
                continue

        if len(X_list) < 5: 
            logger.warning(f"[{self.name}] 有效训练样本仅 {len(X_list)} 个，训练失败。")
            return
        
        X = np.array(X_list)
        X_scaled = self.scaler.fit_transform(X)
        self.probe = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.probe.fit(X_scaled, np.array(y_list))
        self.is_fitted = True
        logger.info(f"[{self.name}] SEP 探针训练完成 (样本: {len(y_list)})")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            feat = self._extract_features(accessor).reshape(1, -1)
            feat_scaled = self.scaler.transform(feat)
            return float(self.probe.predict_proba(feat_scaled)[0][1])
        except Exception:
            return float('nan')