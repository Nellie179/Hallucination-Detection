# baseline_detectors/detectors/sar.py
"""
SAR (Shifting Attention to Relevance) Detector - 相关性加权不确定性检测器

原理：
    并非所有token都同等重要地代表底层含义,语言冗余使得少数关键词就能传达长句的本质。
    SAR通过将注意力转移到更相关的成分(token级和句子级)来进行更好的不确定性量化。

方法：
    1. Token级SAR: 计算每个token的相关性(通过语义相似度),加权token熵
       - RT(zi, s, x) = 1 - |g(x∪s, x∪s\{zi})| (token移除前后的语义相似度)
       - ET(zi, sj, x) = -log p(zi|s<i, x) × R̃T(zi, sj, x) (相关性加权熵)
       - tokenSAR(sj, x) = Σi ET(zi, sj, x)

    2. Sentence级SAR: 计算句子间的相关性,加权句子熵
       - RS(si, S, x) = Σj≠i g(si, sj)p(sj|x) (与其他高概率句子的语义一致性)
       - ES(sj, S, x) = -log(p(sj|x) + (1/t)RS(sj, S, x))
       - sentSAR(S, x) = (1/K) Σk ES(sk, S, x)

    3. 组合SAR: 结合token级和句子级的不确定性
       - SAR分数越低 → 模型越自信 → 幻觉概率越低

参考文献：
    Duan et al. "Shifting Attention to Relevance: Towards the Predictive
    Uncertainty Quantification of Free-Form Large Language Models"
    ACL 2024
    https://aclanthology.org/2024.acl-long.276/
    https://github.com/jinhaoduan/SAR

依赖：
    numpy, torch, transformers (sentence similarity model)
"""

import numpy as np
import logging
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 延迟导入,避免在没有安装transformers时报错
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers未安装,SAR将使用简化版本(基于token概率)")
    SENTENCE_TRANSFORMER_AVAILABLE = False


@register_detector("sar")
class SARDetector(BaseDetector):
    """
    SAR (Shifting Attention to Relevance) 检测器

    需要数据：
        - token_logprobs（token级对数概率）- Blackbox/Whitebox 均可
        - 可选: 多个生成样本（用于句子级SAR）
    """

    def __init__(
            self,
            name: str,
            use_token_level: bool = True,
            use_sentence_level: bool = False,
            similarity_model: str = "all-MiniLM-L6-v2",
            temperature: float = 1.0,
            epsilon: float = 1e-10,
            **kwargs
    ):
        """
        Args:
            use_token_level: 是否使用token级SAR（默认True）
            use_sentence_level: 是否使用sentence级SAR（需要多个生成样本,默认False）
            similarity_model: 用于计算语义相似度的模型（默认使用轻量级模型）
            temperature: 句子级SAR的温度参数（控制相关性转移的尺度）
            epsilon: 用于数值稳定性的小常数
        """
        super().__init__(name, **kwargs)

        self.use_token_level = use_token_level
        self.use_sentence_level = use_sentence_level
        self.temperature = temperature
        self.epsilon = epsilon

        # 初始化语义相似度模型（如果需要）
        self.similarity_model = None
        if SENTENCE_TRANSFORMER_AVAILABLE and (use_token_level or use_sentence_level):
            try:
                self.similarity_model = SentenceTransformer(similarity_model)
                logger.info(f"[{self.name}] 加载语义相似度模型: {similarity_model}")
            except Exception as e:
                logger.warning(f"[{self.name}] 无法加载语义模型: {e}, 将使用简化版本")

        # 统计信息（用于归一化）
        self.sar_mean = None
        self.sar_std = None

        logger.info(f"[{self.name}] SAR检测器初始化完成")
        logger.info(f"  Token级SAR: {use_token_level}")
        logger.info(f"  Sentence级SAR: {use_sentence_level}")
        logger.info(f"  温度参数: {temperature}")

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度 g(text1, text2)

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数 [0, 1]
        """
        if self.similarity_model is not None:
            try:
                embeddings = self.similarity_model.encode([text1, text2])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                # 转换到 [0, 1] 范围
                return (similarity + 1.0) / 2.0
            except Exception as e:
                logger.warning(f"语义相似度计算失败: {e}")
                return 0.5
        else:
            # 简化版本: 使用Jaccard相似度
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            if not tokens1 or not tokens2:
                return 0.0
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.0

    def _compute_token_relevance(
            self,
            tokens: List[str],
            token_idx: int,
            full_text: str,
            prompt: str = ""
    ) -> float:
        """
        计算token的相关性 RT(zi, s, x)
        RT(zi, s, x) = 1 - |g(x∪s, x∪s\{zi})|

        Args:
            tokens: token列表
            token_idx: 当前token的索引
            full_text: 完整文本
            prompt: 输入提示

        Returns:
            相关性分数 [0, 1]
        """
        # 构造移除当前token后的文本
        tokens_without_current = tokens[:token_idx] + tokens[token_idx + 1:]
        text_without_token = " ".join(tokens_without_current)

        # 如果有prompt,与prompt拼接
        if prompt:
            full_with_prompt = prompt + " " + full_text
            without_token_with_prompt = prompt + " " + text_without_token
        else:
            full_with_prompt = full_text
            without_token_with_prompt = text_without_token

        # 计算移除token前后的语义相似度
        similarity = self._compute_semantic_similarity(
            full_with_prompt,
            without_token_with_prompt
        )

        # RT = 1 - similarity (相似度越低,说明token越重要)
        relevance = 1.0 - abs(similarity)

        return max(0.0, min(1.0, relevance))  # 确保在[0,1]范围内

    def _compute_token_level_sar(
            self,
            tokens: List[str],
            token_logprobs: np.ndarray,
            full_text: str,
            prompt: str = ""
    ) -> float:
        """
        计算Token级SAR
        tokenSAR(sj, x) = Σi ET(zi, sj, x)
        其中 ET(zi, sj, x) = -log p(zi|s<i, x) × R̃T(zi, sj, x)

        Args:
            tokens: token列表
            token_logprobs: token对数概率数组
            full_text: 完整文本
            prompt: 输入提示

        Returns:
            Token级SAR分数
        """
        if len(tokens) == 0 or len(token_logprobs) == 0:
            return 0.0

        # 1. 计算每个token的相关性 RT
        relevances = []
        for i in range(len(tokens)):
            relevance = self._compute_token_relevance(tokens, i, full_text, prompt)
            relevances.append(relevance)

        relevances = np.array(relevances)

        # 2. 归一化相关性 R̃T (使得在句子内可比较)
        relevance_sum = np.sum(relevances)
        if relevance_sum < self.epsilon:
            # 所有token相关性都很低,使用均匀权重
            normalized_relevances = np.ones(len(relevances)) / len(relevances)
        else:
            normalized_relevances = relevances / relevance_sum

        # 3. 计算加权token熵 ET
        # ET(zi) = -log p(zi) × R̃T(zi)
        # 注意: token_logprobs 已经是 log概率, 所以 -log p = -logprob
        token_entropies = -token_logprobs * normalized_relevances

        # 4. 求和得到token级SAR
        token_sar = np.sum(token_entropies)

        return float(token_sar)

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """
        计算训练集上的SAR统计信息（用于归一化幻觉分数）
        """
        logger.info(f"[{self.name}] 开始计算训练集SAR统计...")

        sar_scores = []
        for accessor in train_accessors:
            try:
                score = self._compute_sar_score(accessor)
                if not np.isnan(score) and not np.isinf(score):
                    sar_scores.append(score)
            except Exception as e:
                logger.warning(f"Sample {accessor.sample_id}: SAR计算失败: {e}")
                continue

        if not sar_scores:
            logger.warning(f"[{self.name}] 训练集没有有效的SAR数据，使用默认归一化参数")
            self.sar_mean = 5.0
            self.sar_std = 2.0
        else:
            self.sar_mean = np.mean(sar_scores)
            self.sar_std = np.std(sar_scores)
            if self.sar_std < self.epsilon:
                self.sar_std = 1.0  # 避免除零

            logger.info(f"[{self.name}] SAR统计:")
            logger.info(f"  均值: {self.sar_mean:.4f}")
            logger.info(f"  标准差: {self.sar_std:.4f}")
            logger.info(f"  最小值: {np.min(sar_scores):.4f}")
            logger.info(f"  最大值: {np.max(sar_scores):.4f}")

    def _compute_sar_score(self, accessor: SampleAccessor) -> float:
        """
        计算原始SAR分数（未归一化）

        Returns:
            SAR分数（越高表示越不确定,幻觉可能性越大）
        """
        # 获取tokens和对数概率
        tokens = accessor.get_tokens()
        token_logprobs = accessor.get_token_logprobs()

        if len(tokens) == 0 or len(token_logprobs) == 0:
            return 0.0

        # 获取完整文本和提示
        full_text = accessor.get_answer()
        prompt = accessor.get_question() if hasattr(accessor, 'get_question') else ""

        # 只使用token级SAR（简化版本）
        if self.use_token_level:
            sar_score = self._compute_token_level_sar(
                tokens, token_logprobs, full_text, prompt
            )
        else:
            # 如果不使用token级,使用简单的平均负对数概率
            sar_score = -np.mean(token_logprobs)

        return sar_score

    def predict_score(self, accessor: SampleAccessor) -> float:
        """
        计算幻觉分数

        返回值：
            float: 幻觉概率 [0, 1]
                   SAR分数越高 → 不确定性越高 → 幻觉概率越高
        """
        try:
            # 计算SAR分数
            sar_score = self._compute_sar_score(accessor)

            if np.isnan(sar_score) or np.isinf(sar_score):
                logger.warning(f"Sample {accessor.sample_id}: SAR计算结果无效")
                return float('nan')

            # 归一化SAR分数到幻觉概率 [0, 1]
            # SAR分数越高，幻觉分数越高
            if self.sar_mean is not None and self.sar_std is not None:
                # Z-score 归一化，然后映射到 [0, 1]
                z_score = (sar_score - self.sar_mean) / self.sar_std
                # 使用 sigmoid 映射：高SAR → 高分数
                hallucination_score = 1.0 / (1.0 + np.exp(-z_score))
            else:
                # 未训练的情况，使用简单映射
                # 假设SAR分数通常在 0-10 之间
                hallucination_score = min(sar_score / 10.0, 1.0)

            return float(hallucination_score)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: SAR 计算失败: {e}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        """详细分析（调试用）"""
        try:
            tokens = accessor.get_tokens()
            token_logprobs = accessor.get_token_logprobs()
            full_text = accessor.get_answer()

            sar_score = self._compute_sar_score(accessor)

            # 计算token相关性分布
            relevances = []
            for i in range(min(len(tokens), 10)):  # 只分析前10个token
                relevance = self._compute_token_relevance(
                    tokens, i, full_text, ""
                )
                relevances.append(relevance)

            return {
                "num_tokens": len(tokens),
                "sar_score": float(sar_score),
                "sar_mean": float(self.sar_mean) if self.sar_mean is not None else None,
                "sar_std": float(self.sar_std) if self.sar_std is not None else None,
                "avg_token_logprob": float(np.mean(token_logprobs)),
                "token_relevances_sample": relevances,  # 前10个token的相关性
                "tokens_sample": tokens[:10],  # 前10个token
                "hallucination_score": self.predict_score(accessor)
            }

        except Exception as e:
            return {"error": str(e)}


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("SAR (Shifting Attention to Relevance) Detector 单元测试")
    print("=" * 70)

    # 创建模拟数据
    np.random.seed(42)

    class MockAccessor:
        def __init__(self, sample_id, tokens, token_logprobs, answer):
            self.sample_id = sample_id
            self._tokens = tokens
            self._token_logprobs = token_logprobs
            self._answer = answer

        def get_tokens(self):
            return self._tokens

        def get_token_logprobs(self):
            return self._token_logprobs

        def get_answer(self):
            return self._answer

        def get_question(self):
            return "What is the capital of France?"

    # 测试用例 1：高置信度回答（低不确定性，低幻觉）
    print("\n[测试 1] 高置信度生成（期望：低幻觉分数）")
    tokens1 = ["The", "capital", "of", "France", "is", "Paris", "."]
    logprobs1 = np.array([-0.1, -0.2, -0.1, -0.15, -0.1, -0.2, -0.1])  # 高置信度
    answer1 = "The capital of France is Paris."
    accessor1 = MockAccessor("test_001", tokens1, logprobs1, answer1)

    # 测试用例 2：低置信度回答（高不确定性，高幻觉）
    print("\n[测试 2] 低置信度生成（期望：高幻觉分数）")
    tokens2 = ["The", "capital", "might", "possibly", "be", "Rome", "or", "Berlin", "?"]
    logprobs2 = np.array([-0.5, -1.2, -2.5, -2.0, -1.5, -3.0, -2.0, -2.5, -1.0])  # 低置信度
    answer2 = "The capital might possibly be Rome or Berlin?"
    accessor2 = MockAccessor("test_002", tokens2, logprobs2, answer2)

    try:
        detector = SARDetector(
            name="test_sar",
            use_token_level=True,
            use_sentence_level=False
        )

        # 模拟训练（使用两个样本）
        print("\n训练SAR检测器...")
        detector.fit([accessor1, accessor2])

        # 测试 1
        print("\n" + "=" * 70)
        score1 = detector.predict_score(accessor1)
        analysis1 = detector.analyze(accessor1)
        print(f"样本1 - 幻觉分数: {score1:.3f}")
        print(f"  SAR分数: {analysis1['sar_score']:.3f}")
        print(f"  平均log概率: {analysis1['avg_token_logprob']:.3f}")
        print(f"  Tokens: {analysis1['tokens_sample']}")

        # 测试 2
        print("\n" + "=" * 70)
        score2 = detector.predict_score(accessor2)
        analysis2 = detector.analyze(accessor2)
        print(f"样本2 - 幻觉分数: {score2:.3f}")
        print(f"  SAR分数: {analysis2['sar_score']:.3f}")
        print(f"  平均log概率: {analysis2['avg_token_logprob']:.3f}")
        print(f"  Tokens: {analysis2['tokens_sample']}")

        print("\n" + "=" * 70)
        print("✅ 测试完成")
        print(f"预期：高置信度样本分数 ({score1:.3f}) < 低置信度样本分数 ({score2:.3f})")

        if score1 < score2:
            print("✓ 结果符合预期！")
        else:
            print("✗ 结果不符合预期，可能需要调整参数")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
