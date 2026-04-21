# baseline_detectors/detectors/selfcheck_nli.py
"""
SelfCheckNLI Detector - 灰盒 NLI 一致性检测器

原理：
    1. 将主回答 (Main Output) 拆分成单独的句子。
    2. 将每个句子作为 Hypothesis，多次采样的文本 (Stochastic Samples) 作为 Premise。
    3. 用 NLI 模型判断 Premise 是否能蕴含 (Entail) Hypothesis。
    4. 如果多次采样都无法蕴含该句子，说明该句子是幻觉。
    🎯 已对齐官方实现：直接抓取 CONTRADICTION 类的连续 Softmax 概率作为幻觉分数。
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

# 关闭 httpx 刷屏
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

@register_detector("selfcheck_nli")
class SelfCheckNLIDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            nli_model: str = "roberta-large-mnli",  # 🎯 核心修复：换成绝对不会报错的 roberta
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # 声明依赖：需要多次采样文本
        self.requires_stochastic = True
        
        self.nli_model_name = nli_model
        self.device = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.nli_pipeline = None
        self.nlp = None

        logger.info(f"[{self.name}] SelfCheckNLI 初始化完成 (裁判模型: {self.nli_model_name})")

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_dependencies(self):
        """安全加载 NLI 判定模型和 Spacy"""
        if self.nli_pipeline is not None and self.nlp is not None:
            return
        try:
            import spacy
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("未找到 spacy 模型，尝试自动下载...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            logger.info(f"[{self.name}] 正在加载 NLI 模型: {self.nli_model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=16,
                truncation=True,
                max_length=512,
                top_k=None # 🛠️ [官方对齐修改]: 返回所有类的概率分布，而不是单一标签
            )
            logger.info(f"[{self.name}] ✓ NLI 模型加载完成")
        except ImportError as e:
            raise ImportError(f"[{self.name}] 缺少依赖库: {e}")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        """预热加载模型"""
        self._load_dependencies()

    def predict_score(self, accessor: SampleAccessor) -> float:
        if self.nli_pipeline is None or self.nlp is None:
            self._load_dependencies()

        # 1. 获取基础数据
        main_text = accessor.metadata.get("model_output_text", "").strip()
        stochastic_data = accessor.stochastic_samples_dict.get(accessor.sample_id, {})
        samples = stochastic_data.get("samples", []) if isinstance(stochastic_data, dict) else stochastic_data
        valid_samples = [s for s in samples if s and s.strip()]

        if not main_text or not valid_samples:
            return float('nan')

        # 2. 🚀 [核心兼容逻辑]: 智能映射器
        # 尝试从元数据中获取选项（仅多选题型会有此字段）
        choices = accessor.metadata.get("structured_data", {}).get("choices", {})

        def smart_expand(text):
            """
            智能展开函数：
            - 如果 text 是选项字母 (如 'A') 且在 choices 字典中 -> 映射为完整文本
            - 否则 (如文本问答 'The weather is cold') -> 原样返回，不进行任何处理
            """
            clean_text = text.strip().upper()
            # 兼容逻辑：长度短且在选项键值中
            if len(clean_text) <= 2 and clean_text in choices:
                return f"The answer is {choices[clean_text]}"
            return text

        try:
            # 3. 对主回答应用智能展开
            expanded_main = smart_expand(main_text)
            
            # 4. 对主回答分句 (使用 Spacy)
            sentences = [sent.text.strip() for sent in self.nlp(expanded_main).sents]
            sentences = [sent for sent in sentences if len(sent) > 0]
            if not sentences:
                return float('nan')

            # 5. 构建 NLI 验证对
            pairs = []
            pair_indices = []
            
            for s_idx, sentence in enumerate(sentences):
                for samp_idx, sample in enumerate(valid_samples):
                    # 🚀 [核心兼容逻辑]: 采样样本也应用智能展开
                    expanded_sample = smart_expand(sample)
                    pairs.append({"text": expanded_sample, "text_pair": sentence})
                    pair_indices.append((s_idx, samp_idx))

            # 4. 批量推理
            # top_k=None 保证返回格式为 [[{'label': 'A', 'score': 0.1}, ...], ...]
            results = self.nli_pipeline(pairs)

            # 5. 解析连续分数：精准抓取 contradiction 类的概率
            sentence_hallucination_scores = np.zeros(len(sentences))
            
            for (s_idx, samp_idx), res_list in zip(pair_indices, results):
                contradiction_prob = 0.0
                
                # 遍历三个类的打分，寻找代表冲突的类
                # RoBERTa 的类名通常是 CONTRADICTION，但以防万一做了宽松匹配
                for class_score in res_list:
                    label = class_score['label'].upper()
                    if 'CONTRADICTION' in label or label == 'LABEL_0': # roberta 的 0 通常是 contradiction
                        contradiction_prob = class_score['score']
                        break
                
                # 🛠️ [官方对齐修改]: 直接累加矛盾概率，告别离散 0/1 加分
                sentence_hallucination_scores[s_idx] += contradiction_prob

            # 6. 对每个句子取平均（跨多次采样的平均幻觉概率）
            num_samples = len(valid_samples)
            sentence_scores = sentence_hallucination_scores / num_samples

            # 7. 最终得分：整段话的最大句子幻觉得分 (SelfCheck 原论文推荐 max)
            final_score = np.max(sentence_scores)
            
            return float(final_score)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id}: SelfCheckNLI 计算失败: {e}")
            return float('nan')