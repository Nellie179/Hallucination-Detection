# baseline_detectors/detectors/prism.py
"""
PRISM (Prompt-guided Internal States) Detector

原理：
    使用合适的prompts使LLM内部状态中与文本真实性相关的结构
    更加显著和一致,提高跨domain泛化能力。

方法：
    1. 用不同prompts引导生成
    2. 提取prompt-guided internal states
    3. 训练hallucination detector (基于SAPLMA/MIND等)
    4. 跨domain评估

基于官方实现：
    https://github.com/fujie-math/PRISM

参考文献：
    Zhang et al. "Prompt-Guided Internal States for Hallucination Detection
    of Large Language Models"
    ACL 2025
"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_detector("prism")
class PRISMDetector(BaseDetector):
    """
    PRISM检测器
    """

    def __init__(
        self,
        name: str,
        target_layer: int = -1,
        use_prompt_ensemble: bool = False,  # 是否使用prompt集成
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # 🙋‍♂️ 核心添加：向大管家声明我们需要 QA 特征文件
        self.requires_qa_features = True

        self.target_layer = target_layer
        self.use_prompt_ensemble = use_prompt_ensemble

        # 简化版本: 使用线性probe
        self.scaler = StandardScaler()
        self.probe = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.is_fitted = False

        logger.info(f"[{self.name}] PRISM检测器初始化完成")
        logger.info(f"  Prompt集成: {use_prompt_ensemble}")

    def _extract_hidden_state(self, accessor: SampleAccessor) -> np.ndarray:
        """从 Runner 挂载的 H5 句柄中极其安全地提取 PRISM 特征"""
        if getattr(accessor, "qa_h5_file", None) is None:
            raise ValueError(f"样本 {accessor.sample_id} 缺少 QA 特征！请确保 Runner 已挂载特征文件。")
            
        grp_name = f"{accessor.sample_id}_prism"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"H5 文件中缺失样本 {accessor.sample_id} 的 PRISM 组。")
            
        grp = accessor.qa_h5_file[grp_name]
        
        # 智能识别层号 (-1 取最后一层)
        layer_str = f"layer_{self.target_layer}"
        if self.target_layer < 0:
            layers = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("layer_")]
            max_layer = max(layers)
            layer_str = f"layer_{max_layer}"

        # 提取池化后的隐藏层向量
        feat = np.array(grp[layer_str], dtype=np.float32)
        return feat

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] 开始在 Train Set 训练PRISM...")
        X_train, y_train = [], []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]: continue
                X_train.append(self._extract_hidden_state(accessor))
                y_train.append(1 if category == "hallucination" else 0)
            except Exception:
                continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.probe.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        train_acc = self.probe.score(X_train_scaled, y_train)
        logger.info(f"[{self.name}] PRISM训练完成 (Train拟合准确率: {train_acc * 100:.2f}%)")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            # 现在，这里的预测是正儿八经对“从未见过”的 Test 样本进行的！
            feature = self._extract_hidden_state(accessor)
            feature_scaled = self.scaler.transform(feature.reshape(1, -1))
            return float(self.probe.predict_proba(feature_scaled)[0, 1])
        except Exception as e:
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """详细分析"""
        try:
            feature = self._extract_hidden_state(accessor)

            return {
                "feature_dim": len(feature),
                "use_prompt_ensemble": self.use_prompt_ensemble,
                "prompt_guided": True,
                "hallucination_score": self.predict_score(accessor) if self.is_fitted else None
            }
        except Exception as e:
            return {"error": str(e)}