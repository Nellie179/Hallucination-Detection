# baseline_detectors/detectors/icr_probe.py
"""
ICR Probe (Internal Conflict Resolution) Detector
基于 ACL 2025 论文实现。
该探测器利用 MLP 识别模型内部注意力与隐藏层状态之间的冲突信号。
"""

import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 🧠 核心模型定义 (源自你的 icr_probe.py / utils.py)
# ==========================================
class ICR_MLP(nn.Module):
    def __init__(self, input_dim):
        super(ICR_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

@register_detector("icr_probe")
class ICRProbeDetector(BaseDetector):
    def __init__(
        self,
        name: str = "icr_probe",
        use_icr_only: bool = True,  # True 使用 JS 散度向量, False 使用池化隐藏层
        learning_rate: float = 1e-3,
        epochs: int = 15,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # 🙋‍♂️ 关键声明：需要大管家挂载 QA 特征文件
        self.requires_qa_features = True
        
        self.use_icr_only = use_icr_only
        self.lr = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.is_fitted = False

        logger.info(f"[{self.name}] ICR Probe 初始化完成 (模式: {'JS-Divergence' if use_icr_only else 'Mean-Pooling HS'})")

    def _get_feature(self, accessor: SampleAccessor) -> np.ndarray:
        """从 H5 中提取特征：优先取 JS 散度向量"""
        if accessor.qa_h5_file is None:
            raise ValueError(f"样本 {accessor.sample_id} 缺失 H5 句柄")
            
        grp_name = f"{accessor.sample_id}_icr_probe"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"H5 中缺失样本 {accessor.sample_id} 的 ICR 数据组")
            
        grp = accessor.qa_h5_file[grp_name]
        
        if self.use_icr_only:
            # 🎯 提取由 extract_qa_hidden_states.py 算好的 (L,) 维 JS 向量
            if "icr_feature" not in grp:
                raise KeyError(f"样本 {accessor.sample_id} 尚未提取 icr_feature！")
            return np.array(grp["icr_feature"], dtype=np.float32)
        else:
            # 备选方案：提取池化的隐藏层 (类似 SAPLMA)
            # 取最后一层作为示例
            layers = [k for k in grp.keys() if k.startswith("layer_")]
            last_layer = sorted(layers, key=lambda x: int(x.split("_")[1]))[-1]
            return np.array(grp[last_layer], dtype=np.float32)

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """训练 ICR MLP 探测器"""
        logger.info(f"[{self.name}] 开始训练 MLP 网络...")
        
        X_train, y_train = [], []
        for acc in train_accessors:
            try:
                category = acc.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue
                
                feat = self._get_feature(acc)
                label = 1 if category == "hallucination" else 0
                
                X_train.append(feat)
                y_train.append(label)
            except Exception as e:
                continue

        if len(set(y_train)) < 2:
            logger.warning(f"[{self.name}] 训练集标签不足，跳过训练。")
            return

        X = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1).to(self.device)

        # 初始化并训练模型
        self.model = ICR_MLP(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        self.is_fitted = True
        logger.info(f"[{self.name}] 训练完成 (Epochs: {self.epochs})")

    def predict_score(self, accessor: SampleAccessor) -> float:
        """预测幻觉概率"""
        if not self.is_fitted or self.model is None:
            return float('nan')

        try:
            feat = self._get_feature(accessor)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                prob = self.model(x).cpu().item()
            return float(prob)
            
        except Exception as e:
            logger.error(f"样本 {accessor.sample_id} 预测失败: {e}")
            return float('nan')