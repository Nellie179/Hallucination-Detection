"""
SelfCheckGPT (BERTScore variant) - 基于 BERTScore 的自我一致性检测

全新架构版：
    - 零外部路径依赖 (Zero-Configuration for I/O)
    - 通过 requires_stochastic = True 向主调度器 (Runner) 声明数据依赖
    - 纯粹的算法实现，只与 SampleAccessor 交互
"""

import numpy as np
import logging
from typing import List
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_detector("selfcheck_bertscore")
class SelfCheckBERTScoreDetector(BaseDetector):
    """
    SelfCheckGPT - BERTScore 变体
    """

    def __init__(
            self,
            name: str,
            bert_model: str = "roberta-large",
            device: str = None,
            **kwargs
    ):
        """
        干净的初始化：只保留算法超参数，彻底踢掉文件路径和生成模型的配置！
        """
        super().__init__(name, **kwargs)

        # 🙋‍♂️ 核心魔法：向 Runner 举手声明依赖！
        # Runner 看到这个标志，就会在跑之前乖乖把 Stochastic 采样准备好。
        self.requires_stochastic = True
        
        self.bert_model = bert_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")

        # 延迟加载 BERTScore 模型（避免初始化时占用资源）
        self.scorer = None

        logger.info(f"[{self.name}] SelfCheckGPT(BERTScore) 初始化完成")
        logger.info(f"  - 依赖声明: requires_stochastic = True")
        logger.info(f"  - BERT 模型: {bert_model}")
        logger.info(f"  - 设备: {self.device}")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_scorer(self):
        """延迟加载 BERTScore 模型"""
        if self.scorer is not None:
            return

        try:
            from bert_score import BERTScorer

            logger.info(f"[{self.name}] 正在加载 BERTScore 模型: {self.bert_model}...")

            self.scorer = BERTScorer(
                model_type=self.bert_model,
                lang="en",
                rescale_with_baseline=False,  # 🛠️ [核心修复]: 彻底关闭基线缩放，防止单字符越界崩溃
                device=self.device
            )

            logger.info(f"[{self.name}] ✓ BERTScore 模型加载完成")

        except ImportError:
            raise ImportError(
                f"[{self.name}] 缺少依赖库 bert-score\n"
                f"请安装: pip install bert-score"
            )
        except Exception as e:
            raise RuntimeError(f"[{self.name}] BERTScore 模型加载失败: {e}")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        免训练方法，仅用于在开始评估前把 BERT 模型加载进显存
        """
        self._load_scorer()

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        核心评估逻辑：计算主答案与多次采样的 BERTScore F1 分数。
        """
        if self.scorer is None:
            self._load_scorer()

        # 优雅地从管家那里调取对应样本的数据
        main_output = accessor.get_model_output_text()
        samples = accessor.get_stochastic_samples()

        # 鲁棒性检查
        if not main_output or not main_output.strip():
            logger.debug(f"Sample {accessor.sample_id}: 主输出为空")
            return float('nan')

        if not samples:
            logger.warning(f"Sample {accessor.sample_id}: 缺少采样数据")
            return float('nan')

        valid_samples = [s for s in samples if s and s.strip()]
        if not valid_samples:
            return float('nan')

        try:
            # BERTScore 计算：输入格式为 (cands, refs)
            # 我们将 valid_samples 作为候选 (hypotheses/cands)
            # 主答案复制多份作为参考 (references)
            references = [main_output] * len(valid_samples)
            hypotheses = valid_samples

            P, R, F1 = self.scorer.score(hypotheses, references)

            # 使用 F1 分数的平均值作为一致性度量
            avg_f1 = F1.mean().item()

            # 转换为幻觉分数：相似度越低，幻觉概率越高
            hallucination_score = 1.0 - avg_f1

            return float(hallucination_score)

        except Exception as e:
            # 🛠️ [核心修复]: 拒绝盲猜，保留完整堆栈日志
            logger.error(f"Sample {accessor.sample_id}: BERTScore 计算失败:\n{traceback.format_exc()}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """
        详细分析（用于调试和可视化）
        """
        if self.scorer is None:
            self._load_scorer()

        main_output = accessor.get_model_output_text()
        samples = accessor.get_stochastic_samples()

        valid_samples = [s for s in samples if s and s.strip()]

        if not valid_samples:
            return {"error": "No valid samples"}

        references = [main_output] * len(valid_samples)
        P, R, F1 = self.scorer.score(valid_samples, references)

        f1_scores = F1.tolist()

        return {
            "main_output": main_output,
            "num_samples": len(valid_samples),
            "f1_scores": f1_scores,
            "avg_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "min_f1": float(np.min(f1_scores)),
            "max_f1": float(np.max(f1_scores)),
            "hallucination_score": 1.0 - np.mean(f1_scores)
        }


# ==========================================
# 单元测试 (纯净离线版)
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("SelfCheckGPT (BERTScore) 单元测试")
    print("=" * 70)

    # 模拟 Accessor，脱离文件系统测试算法逻辑
    class MockAccessor(SampleAccessor):
        def __init__(self, sample_id, main_output, samples):
            self.sample_id = sample_id
            self.metadata = {"model_output_text": main_output}
            self.h5_group = None
            self.stochastic_samples_dict = {sample_id: samples}

    # 测试用例 1：高一致性（非幻觉）
    print("\n[测试 1] 高一致性答案（期望：低幻觉分数）")
    accessor1 = MockAccessor(
        sample_id="test_001",
        main_output="Paris is the capital of France.",
        samples=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "Paris serves as the capital city of France.",
            "France's capital is Paris.",
        ]
    )

    # 测试用例 2：低一致性（可能幻觉）
    print("\n[测试 2] 低一致性答案（期望：高幻觉分数）")
    accessor2 = MockAccessor(
        sample_id="test_002",
        main_output="The capital of France is Lyon.",
        samples=[
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "I think it might be Marseille.",
            "Bordeaux could be the capital.",
        ]
    )

    try:
        # 使用一个较小的 deberta 模型测试，跑得快
        detector = SelfCheckBERTScoreDetector(
            name="test_bertscore",
            bert_model="microsoft/deberta-v3-small"
        )
        
        # 确认已正确声明依赖
        assert detector.requires_stochastic == True

        detector.fit([accessor1, accessor2])

        # 测试 1
        score1 = detector.predict_score(accessor1)
        analysis1 = detector.analyze(accessor1)
        print(f"幻觉分数: {score1:.3f} | 平均 F1: {analysis1['avg_f1']:.3f}")

        # 测试 2
        score2 = detector.predict_score(accessor2)
        analysis2 = detector.analyze(accessor2)
        print(f"幻觉分数: {score2:.3f} | 平均 F1: {analysis2['avg_f1']:.3f}")

        print("\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()