# baseline_detectors/detectors/icr_probe.py
"""
ICR Probe (Information Contribution to Residual Stream) Detector

原理：
    跟踪hidden states在各层之间的动态变化来检测幻觉。
    现有方法主要关注静态和孤立的表示,忽略了它们在各层之间的动态演化。

    ICR Score量化不同模块(FFN或self-attention)对hidden state更新的贡献。
    ICR Probe聚合所有层的ICR Score来捕获residual stream的综合动态。

方法：
    1. 计算ICR Score: 量化各层模块对hidden state更新的贡献
       ICR_l = ||h_l - h_{l-1}|| / ||h_l||

    2. 聚合多层的ICR Score作为特征

    3. 训练一个简单的分类器(线性probe或MLP)来检测幻觉

优势：
    - 捕获动态的forward pass变化模式
    - 对非标准生成错误更敏感
    - 参数少,效率高

参考文献：
    Zhang et al. "ICR Probe: Tracking Hidden State Dynamics for Reliable
    Hallucination Detection in LLMs"
    ACL 2025
    https://arxiv.org/abs/2507.16488

依赖：
    numpy, sklearn (用于probe训练)
"""

import numpy as np
import logging
from typing import List, Dict
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


@register_detector("icr_probe")
class ICRProbeDetector(BaseDetector):
    """
    ICR Probe 检测器

    需要数据：
        - 多层的hidden states - Whitebox方法
        - 可以使用现有的answer hidden states!
    """

    def __init__(
        self,
        name: str,
        layer_range: tuple = None,  # (start, end) 或 None表示使用所有可用层
        normalization: str = "l2",  # l2 或 l1 归一化
        aggregation: str = "mean",  # mean, max, concat聚合方式
        epsilon: float = 1e-10,
        **kwargs
    ):
        """
        Args:
            layer_range: 使用的层范围,None表示使用所有层
            normalization: ICR Score的归一化方式
            aggregation: 多层ICR Score的聚合方式
            epsilon: 数值稳定性常数
        """
        super().__init__(name, **kwargs)

        self.layer_range = layer_range
        self.normalization = normalization
        self.aggregation = aggregation
        self.epsilon = epsilon

        # Probe分类器
        self.scaler = StandardScaler()
        self.probe = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.is_fitted = False

        logger.info(f"[{self.name}] ICR Probe检测器初始化完成")
        logger.info(f"  层范围: {layer_range if layer_range else '所有可用层'}")
        logger.info(f"  归一化: {normalization}")
        logger.info(f"  聚合方式: {aggregation}")

    def _compute_icr_scores(
        self,
        accessor: SampleAccessor
    ) -> np.ndarray:
        """
        计算ICR Scores

        ICR_l = ||h_l - h_{l-1}|| / (||h_l|| + epsilon)

        Args:
            accessor: 样本访问器

        Returns:
            ICR scores数组, shape: [num_layers-1]

        Note:
            ICR Probe 需要 Q+A 拼接后的 hidden states
            数据来源: extract_qa_hidden_states.py --method icr_probe
        """
        # 获取所有层的 Q+A hidden states
        all_layers_states = accessor.get_qa_all_layers_hidden_states()

        if not all_layers_states:
            raise ValueError(
                f"样本 {accessor.sample_id} 没有 Q+A 多层 hidden states. "
                "请运行 extract_qa_hidden_states.py --method icr_probe"
            )

        # 获取层索引并排序
        layer_indices = sorted(all_layers_states.keys())

        # 应用层范围过滤
        if self.layer_range:
            start, end = self.layer_range
            layer_indices = [l for l in layer_indices if start <= l < end]

        if len(layer_indices) < 2:
            raise ValueError(f"至少需要2层来计算ICR Score,当前只有{len(layer_indices)}层")

        # 计算每层的ICR Score
        icr_scores = []

        for i in range(1, len(layer_indices)):
            prev_layer = layer_indices[i-1]
            curr_layer = layer_indices[i]

            h_prev = all_layers_states[prev_layer]  # shape: [hidden_dim]
            h_curr = all_layers_states[curr_layer]

            # 计算变化量
            delta = h_curr - h_prev

            # 归一化
            if self.normalization == "l2":
                delta_norm = np.linalg.norm(delta, ord=2)
                h_curr_norm = np.linalg.norm(h_curr, ord=2)
            elif self.normalization == "l1":
                delta_norm = np.linalg.norm(delta, ord=1)
                h_curr_norm = np.linalg.norm(h_curr, ord=1)
            else:
                raise ValueError(f"不支持的归一化方式: {self.normalization}")

            # ICR Score = ||delta|| / ||h_curr||
            icr = delta_norm / (h_curr_norm + self.epsilon)
            icr_scores.append(icr)

        return np.array(icr_scores)

    def _aggregate_icr_scores(self, icr_scores: np.ndarray) -> np.ndarray:
        """
        聚合ICR Scores作为特征

        Args:
            icr_scores: ICR scores, shape: [num_layers-1]

        Returns:
            聚合后的特征向量
        """
        if self.aggregation == "mean":
            # 使用均值和标准差作为特征
            features = np.array([
                np.mean(icr_scores),
                np.std(icr_scores),
                np.min(icr_scores),
                np.max(icr_scores)
            ])
        elif self.aggregation == "max":
            # 使用最大值相关统计
            features = np.array([
                np.max(icr_scores),
                np.argmax(icr_scores) / len(icr_scores),  # 归一化位置
                np.mean(icr_scores),
                np.std(icr_scores)
            ])
        elif self.aggregation == "concat":
            # 直接拼接所有ICR scores
            features = icr_scores
        else:
            raise ValueError(f"不支持的聚合方式: {self.aggregation}")

        return features

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        训练ICR Probe
        """
        logger.info(f"[{self.name}] 开始训练ICR Probe...")

        X_train = []
        y_train = []

        for accessor in train_accessors:
            try:
                # 获取标签
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue

                # 计算ICR scores
                icr_scores = self._compute_icr_scores(accessor)

                # 聚合为特征
                features = self._aggregate_icr_scores(icr_scores)

                # 标签: 幻觉=1, 正确=0
                label = 1 if category == "hallucination" else 0

                X_train.append(features)
                y_train.append(label)

            except Exception as e:
                logger.warning(f"样本 {accessor.sample_id} 处理失败: {e}")
                continue

        if len(set(y_train)) < 2:
            raise ValueError(f"训练集必须同时包含correct和hallucination样本!")

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        logger.info(f"[{self.name}] 训练样本数: {len(X_train)}")
        logger.info(f"  Correct: {np.sum(y_train == 0)}")
        logger.info(f"  Hallucination: {np.sum(y_train == 1)}")
        logger.info(f"  特征维度: {X_train.shape[1]}")

        # 标准化和训练
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.probe.fit(X_train_scaled, y_train)
        self.is_fitted = True

        logger.info(f"[{self.name}] ICR Probe训练完成")

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        预测幻觉分数

        Returns:
            float: 幻觉概率 [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] Probe尚未训练")

        try:
            # 计算ICR scores
            icr_scores = self._compute_icr_scores(accessor)

            # 聚合特征
            features = self._aggregate_icr_scores(icr_scores)

            # 预测
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prob_hallucination = self.probe.predict_proba(features_scaled)[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"样本 {accessor.sample_id} 预测失败: {e}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """详细分析(调试用)"""
        try:
            icr_scores = self._compute_icr_scores(accessor)
            features = self._aggregate_icr_scores(icr_scores)

            return {
                "icr_scores": icr_scores.tolist(),
                "icr_mean": float(np.mean(icr_scores)),
                "icr_std": float(np.std(icr_scores)),
                "icr_max": float(np.max(icr_scores)),
                "features": features.tolist(),
                "hallucination_score": self.predict_score(accessor) if self.is_fitted else None
            }
        except Exception as e:
            return {"error": str(e)}


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("ICR Probe Detector 单元测试")
    print("=" * 70)

    np.random.seed(42)

    class MockAccessor:
        def __init__(self, sample_id, layer_states_dict, category):
            self.sample_id = sample_id
            self.layer_states = layer_states_dict
            self.metadata = {"eval_category": category}

        def get_all_layers_hidden_states(self) -> Dict[int, np.ndarray]:
            return self.layer_states

    # 测试用例1: 正常生成(small changes)
    print("\n[测试 1] 正常生成(期望: 低幻觉分数)")
    base_state = np.random.randn(128)
    layers1 = {}
    for i in range(16, 24):
        # 小变化
        layers1[i] = base_state + 0.1 * i * np.random.randn(128)
    accessor1 = MockAccessor("test_001", layers1, "correct")

    # 测试用例2: 幻觉生成(large changes)
    print("\n[测试 2] 幻觉生成(期望: 高幻觉分数)")
    layers2 = {}
    for i in range(16, 24):
        # 大变化
        layers2[i] = np.random.randn(128) + i
    accessor2 = MockAccessor("test_002", layers2, "hallucination")

    try:
        detector = ICRProbeDetector(
            name="test_icr",
            layer_range=None,
            aggregation="mean"
        )

        # 训练
        print("\n训练ICR Probe...")
        detector.fit([accessor1, accessor2])

        # 测试1
        print("\n" + "=" * 70)
        score1 = detector.predict_score(accessor1)
        analysis1 = detector.analyze(accessor1)
        print(f"样本1 - 幻觉分数: {score1:.3f}")
        print(f"  ICR均值: {analysis1['icr_mean']:.4f}")
        print(f"  ICR标准差: {analysis1['icr_std']:.4f}")

        # 测试2
        print("\n" + "=" * 70)
        score2 = detector.predict_score(accessor2)
        analysis2 = detector.analyze(accessor2)
        print(f"样本2 - 幻觉分数: {score2:.3f}")
        print(f"  ICR均值: {analysis2['icr_mean']:.4f}")
        print(f"  ICR标准差: {analysis2['icr_std']:.4f}")

        print("\n" + "=" * 70)
        print("✅ 测试完成")
        print(f"预期: 正常样本分数 ({score1:.3f}) < 幻觉样本分数 ({score2:.3f})")

        if score1 < score2:
            print("✓ 结果符合预期!")
        else:
            print("✗ 结果可能不符合预期(需要更多训练数据)")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
