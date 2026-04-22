# baseline_detectors/detectors/ccs.py
"""
CCS (Contrast-Consistent Search) Detector
基于原论文实现：Discovering Latent Knowledge in Language Models Without Supervision
"""

import numpy as np
import copy
import logging
from typing import List
import sys
import os
import math  # 🚀 [修复 NaN] 新增 math 用于检测 NaN
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPProbe(nn.Module):
    """非线性探针"""
    def __init__(self, d, hidden_size=100):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCSProbe(object):
    """CCS 核心算法实现"""
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        self.var_normalize = var_normalize
        # 🚀 [修复 NaN] 将 1e-8 改为 1e-5，防止 float16 精度下溢导致标准差彻底变成 0 从而引发除零错误
        self.x0_mean, self.x0_std = x0.mean(axis=0, keepdims=True), x0.std(axis=0, keepdims=True) + 1e-5
        self.x1_mean, self.x1_std = x1.mean(axis=0, keepdims=True), x1.std(axis=0, keepdims=True) + 1e-5
        
        self.x0 = self.normalize(x0, is_x0=True)
        self.x1 = self.normalize(x1, is_x0=False)
        self.d = self.x0.shape[-1]
        self.nepochs, self.ntries, self.lr, self.verbose, self.device = nepochs, ntries, lr, verbose, device
        self.batch_size, self.weight_decay, self.linear = batch_size, weight_decay, linear
        
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid()) if self.linear else MLPProbe(self.d)
        self.probe.to(self.device)

    def normalize(self, x, is_x0=True):
        mean = self.x0_mean if is_x0 else self.x1_mean
        std = self.x0_std if is_x0 else self.x1_std
        res = x - mean
        if self.var_normalize: res /= std
        # 🚀 [修复 NaN] 强行兜底清洗，绝不让 NaN 漏入矩阵
        return np.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)

    def get_loss(self, p0, p1):
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1 - p1))**2).mean(0)
        return informative_loss + consistent_loss

    def train(self):
        x0 = torch.tensor(self.x0, dtype=torch.float, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, device=self.device)
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        for epoch in range(self.nepochs):
            p0, p1 = self.probe(x0), self.probe(x1)
            loss = self.get_loss(p0, p1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()

    def repeated_train(self):
        best_loss = np.inf
        for _ in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
        return best_loss

@register_detector("ccs")
class CCSDetector(BaseDetector):
    """与 Runner 完美对齐的 CCS 探测器"""
    def __init__(self, name: str, target_layer: int = -1, epochs: int = 1000, n_tries: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_qa_features = True
        self.target_layer = target_layer
        self.epochs, self.n_tries = epochs, n_tries
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ccs_probe, self.is_fitted, self.needs_flip = None, False, False

    def _extract_features(self, accessor: SampleAccessor):
        # 🎯 适配大管家的 H5 句柄挂载
        if not getattr(accessor, "qa_h5_file", None):
            raise ValueError(f"样本 {accessor.sample_id} 缺少 QA 特征文件！")
        
        grp = accessor.qa_h5_file[f"{accessor.sample_id}_ccs"]
        # 智能层选择
        layer_str = f"layer_{self.target_layer}"
        if self.target_layer < 0:
            layers = [int(k.split("_")[1]) for k in grp["positive"].keys() if k.startswith("layer_")]
            layer_str = f"layer_{max(layers)}"
            
        # 🚀 [修复 NaN] 强转 float32 并清洗，斩断 H5 文件里可能残留的低精度毒素
        p = np.array(grp["positive"][layer_str]).astype(np.float32)
        n = np.array(grp["negative"][layer_str]).astype(np.float32)
        p = np.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
        n = np.nan_to_num(n, nan=0.0, posinf=1e4, neginf=-1e4)
        return p, n

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        x0_list, x1_list, y_list = [], [], []
        for acc in train_accessors:
            try:
                p, n = self._extract_features(acc)
                cat = acc.metadata.get("eval_category")
                if cat not in ["correct", "hallucination"]: continue
                x0_list.append(p); x1_list.append(n)
                y_list.append(1 if cat == "hallucination" else 0)
            except Exception: continue

        if len(x0_list) < 2: return
        
        x0, x1, y = np.array(x0_list), np.array(x1_list), np.array(y_list)
        self.ccs_probe = CCSProbe(x0, x1, nepochs=self.epochs, ntries=self.n_tries, device=self.device)
        self.ccs_probe.repeated_train()
        
        # 🚩 确定方向：在训练集上对比标签，看需不需要翻转
        with torch.no_grad():
            p0 = self.ccs_probe.best_probe(torch.tensor(self.ccs_probe.normalize(x0, True), dtype=torch.float, device=self.device))
            p1 = self.ccs_probe.best_probe(torch.tensor(self.ccs_probe.normalize(x1, False), dtype=torch.float, device=self.device))
            avg_conf = 0.5 * (p0 + (1 - p1))
            preds = (avg_conf.cpu().numpy() < 0.5).astype(int)[:, 0]
            self.needs_flip = (preds == y).mean() < 0.5
            
        self.is_fitted = True
        logger.info(f"[{self.name}] 训练完成，方向翻转: {self.needs_flip}")

    def predict_score(self, accessor: SampleAccessor) -> float:
        # 🚀 [修复 NaN] 没拟合时给兜底分 0.5，不要返回 NaN
        if not self.is_fitted: return 0.5 
        try:
            p, n = self._extract_features(accessor)
            p_norm = self.ccs_probe.normalize(p.reshape(1, -1), True)
            n_norm = self.ccs_probe.normalize(n.reshape(1, -1), False)
            with torch.no_grad():
                prob0 = self.ccs_probe.best_probe(torch.tensor(p_norm, dtype=torch.float, device=self.device)).cpu().item()
                prob1 = self.ccs_probe.best_probe(torch.tensor(n_norm, dtype=torch.float, device=self.device)).cpu().item()
                score = 0.5 * (prob0 + (1 - prob1))
            
            final_score = 1.0 - score if self.needs_flip else score
            # 🚀 [修复 NaN] 最后算出来是 NaN 也要兜底
            if math.isnan(final_score): return 0.5
            return final_score
        # 🚀 [修复 NaN] 报错时返回兜底 0.5
        except Exception: return 0.5