# baseline_detectors/detectors/base.py
from typing import List, Optional
import sys
import os

# 添加父目录到路径以导入 data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_utils.accessor import SampleAccessor


class BaseDetector:
    """
    所有幻觉检测器的基类。

    新版接口设计：
    - 使用 SampleAccessor 统一封装数据访问
    - 简化 detector 实现，无需直接操作 H5 和 metadata
    - 【新增】引入数据依赖声明机制，配合 Runner 实现按需生成 (On-Demand Generation)
    """
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

        # ==========================================
        # 🚀 数据依赖声明 (Dependency Declaration)
        # ==========================================
        # 默认情况下，探测器只需要阶段一的基础数据（metadata 和原生的 hidden states）。
        # 如果具体的子类检测器需要额外数据，必须在它自己的 __init__ 中将对应的标识设为 True！
        
        # 1. 是否需要多次随机采样文本 (如 SelfCheckGPT 需要)
        self.requires_stochastic: bool = False
        
        # 2. 是否需要 Q+A 拼接后的二次推理隐藏状态 (如 CCS, PRISM 需要)
        self.requires_qa_features: bool = False
        
        # 3. 如果需要 QA 特征，具体是哪种构造方法？(可选: 'ccs', 'prism', 'icr_probe' 等)
        self.required_qa_method: Optional[str] = None

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        训练接口（针对 Learning-based 方法，如 Linear Probe）。

        Args:
            train_accessors: 训练集的 SampleAccessor 列表

        免训练方法（如 Entropy, SelfCheckGPT 等）可保持空实现。
        """
        pass

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        推理接口。

        Args:
            accessor: 单个样本的 SampleAccessor 对象

        Returns:
            该样本为幻觉的概率分数（0-1之间，越高越可能是幻觉）
        """
        raise NotImplementedError