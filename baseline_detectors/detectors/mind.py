# baseline_detectors/detectors/mind.py
"""
MIND (Multi-task INternal Detection) Detector - 多任务内部状态检测器

原理：
    无监督实时幻觉检测,基于LLM的内部状态。
    从Wikipedia自动提取伪训练数据,无需人工标注。
    使用简单的MLP对内部状态进行分类。

方法：
    1. 特征提取:
       - 严格对齐官方源码: 提取最后层最后一个Token的hidden state，与最后层全局平均hidden state拼接。
       - 不使用对数概率和熵。

    2. MLP架构 (对齐官方):
       - 输入层首先应用 Dropout (0.2)
       - 4层: [input_dim] -> 256 -> 128 -> 64 -> 2
       - 中间使用 ReLU 激活

    3. 训练:
       - 使用Wikipedia生成伪标注数据(可选)
       - 或使用已有的标注数据

优势：
    - 无需人工标注
    - 实时检测(单次forward pass)
    - 简单高效的MLP架构

参考文献：
    Su et al. "Unsupervised Real-Time Hallucination Detection based on the
    Internal States of Large Language Models"
    ACL 2024
    https://arxiv.org/abs/2403.06448
    https://github.com/oneal2000/MIND

依赖：
    numpy, torch
"""

import numpy as np
import logging
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入torch,如果没有则使用sklearn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from sklearn.neural_network import MLPClassifier
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装,将使用sklearn的MLP")


class MINDMLPTorch(nn.Module):
    """MIND的MLP分类器(PyTorch版本) - 架构已严格对齐官方源码"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()

        layers = []
        
        # 官方源码：Dropout 放置在输入层之后，而非每层之后
        layers.append(nn.Dropout(dropout))
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
                # 注意：此处没有 Dropout，以对齐官方设定
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@register_detector("mind")
class MINDDetector(BaseDetector):
    """
    MIND检测器

    需要数据：
        - hidden_states (最后层)
    """

    def __init__(
        self,
        name: str,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = None,
        **kwargs
    ):
        """
        Args:
            hidden_dims: MLP隐藏层维度
            dropout: 输入层Dropout率
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批大小
            device: 设备 (cuda/cpu)
        """
        super().__init__(name, **kwargs)

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if TORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None  # 在fit时初始化
            self.use_torch = True
        else:
            self.device = 'cpu'
            # sklearn MLP (如果无Torch的兜底)
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_dims),
                activation='relu',
                max_iter=epochs * 10,
                batch_size=batch_size,
                learning_rate_init=learning_rate,
                early_stopping=True,
                n_iter_no_change=5
            )
            self.use_torch = False

        self.is_fitted = False
        self.feature_dim = None

        logger.info(f"[{self.name}] MIND检测器初始化完成")
        logger.info(f"  使用{'PyTorch' if self.use_torch else 'sklearn'}")
        logger.info(f"  隐藏层: {hidden_dims}")
        logger.info(f"  设备: {self.device}")

    def _extract_features(self, accessor: SampleAccessor) -> np.ndarray:
        """
        提取MIND特征:
        官方源码逻辑 -> d["hd_last_token"] + d["hd_last_mean"]
        即最后层最后一个Token的隐藏状态，与全局平均隐藏状态的拼接。
        """
        try:
            # 分别提取 last pooling 和 mean pooling
            hidden_state_last = accessor.get_hidden_states(layer_idx=-1, pooling="last")
            hidden_state_mean = accessor.get_hidden_states(layer_idx=-1, pooling="mean")
            
            # 物理拼接，形成如 8192 维的联合特征
            combined_features = np.concatenate([hidden_state_last, hidden_state_mean])
            
            return combined_features
            
        except Exception as e:
            logger.warning(f"无法获取hidden states: {e}")
            if self.feature_dim:
                return np.zeros(self.feature_dim)
            else:
                raise ValueError("首次提取特征时必须有hidden states")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        训练MIND分类器
        """
        logger.info(f"[{self.name}] 开始训练MIND...")

        # 提取特征和标签
        X_train = []
        y_train = []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue

                features = self._extract_features(accessor)
                label = 1 if category == "hallucination" else 0

                X_train.append(features)
                y_train.append(label)

            except Exception as e:
                logger.warning(f"样本 {accessor.sample_id} 处理失败: {e}")
                continue

        if len(X_train) == 0:
            raise ValueError("没有有效的训练样本!")

        if len(set(y_train)) < 2:
            raise ValueError("训练集必须包含correct和hallucination样本!")

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        self.feature_dim = X_train.shape[1]

        logger.info(f"[{self.name}] 训练样本数: {len(X_train)}")
        logger.info(f"  特征维度: {self.feature_dim}")
        logger.info(f"  Correct: {np.sum(y_train == 0)}")
        logger.info(f"  Hallucination: {np.sum(y_train == 1)}")

        if self.use_torch:
            self._train_torch(X_train, y_train)
        else:
            self._train_sklearn(X_train, y_train)

        self.is_fitted = True
        logger.info(f"[{self.name}] MIND训练完成")

    def _train_torch(self, X_train, y_train):
        """使用PyTorch训练"""
        # 初始化模型
        self.model = MINDMLPTorch(
            input_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        # 转换数据
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 优化器和损失
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()

    def _train_sklearn(self, X_train, y_train):
        """使用sklearn训练"""
        self.model.fit(X_train, y_train)

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        预测幻觉分数

        Returns:
            float: 幻觉概率 [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] 模型尚未训练")

        try:
            features = self._extract_features(accessor)

            if self.use_torch:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    outputs = self.model(X_tensor)
                    probs = F.softmax(outputs, dim=1)
                    prob_hallucination = probs[0, 1].cpu().item()
            else:
                # sklearn
                probs = self.model.predict_proba(features.reshape(1, -1))
                prob_hallucination = probs[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"样本 {accessor.sample_id} 预测失败: {e}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """详细分析(调试用)"""
        try:
            features = self._extract_features(accessor)

            return {
                "feature_dim": len(features),
                "feature_mean": float(np.mean(features)),
                "feature_std": float(np.std(features)),
                "model_type": "PyTorch" if self.use_torch else "sklearn",
                "hallucination_score": self.predict_score(accessor) if self.is_fitted else None
            }
        except Exception as e:
            return {"error": str(e)}


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("MIND Detector 单元测试")
    print("=" * 70)

    np.random.seed(42)

    class MockAccessor:
        def __init__(self, sample_id, hidden_state, category):
            self.sample_id = sample_id
            self.hidden_state = hidden_state
            self.metadata = {"eval_category": category}

        def get_hidden_states(self, layer_idx=-1, pooling="mean"):
            if pooling == "mean":
                return np.mean(self.hidden_state, axis=0)
            elif pooling == "last":
                return self.hidden_state[-1]
            return self.hidden_state[-1]

    # 测试用例1: Correct (高分状态)
    print("\n[测试 1] Correct生成")
    states1 = np.random.randn(10, 128) + 1.0 # 模拟10个token，隐藏维度128
    accessor1 = MockAccessor("test_001", states1, "correct")

    # 测试用例2: Hallucination (低分状态)
    print("\n[测试 2] Hallucination生成")
    states2 = np.random.randn(10, 128) - 1.0
    accessor2 = MockAccessor("test_002", states2, "hallucination")

    try:
        detector = MINDDetector(
            name="test_mind",
            hidden_dims=[128, 64],
            epochs=10
        )

        # 训练
        print("\n训练MIND...")
        train_data = [accessor1, accessor2] * 5  # 重复以增加样本
        detector.fit(train_data)

        # 测试
        print("\n" + "=" * 70)
        score1 = detector.predict_score(accessor1)
        print(f"样本1 (Correct) - 幻觉分数: {score1:.3f}")

        score2 = detector.predict_score(accessor2)
        print(f"样本2 (Hallucination) - 幻觉分数: {score2:.3f}")

        print("\n" + "=" * 70)
        print("✅ 测试完成")
        print(f"预期: Correct分数 ({score1:.3f}) < Hallucination分数 ({score2:.3f})")

        if score1 < score2:
            print("✓ 结果符合预期!")
        else:
            print("✗ 结果可能不符合预期(需要更多训练数据)")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()