import json
import logging
import os
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("tsv")
class TSVDetector(BaseDetector):
    def __init__(self, name="tsv", **kwargs):
        super().__init__(name, **kwargs)
        self.tsv_scores_cache = {}
        self.cache_loaded = False

    def _load_scores_if_needed(self, accessor):
        if self.cache_loaded: return
        
        # 🚀 寻路雷达：直接从大管家传来的基础特征句柄中，逆向解析出实验目录的绝对路径
        base_dir = "."
        if hasattr(accessor, 'h5_group') and accessor.h5_group is not None:
            base_dir = os.path.dirname(accessor.h5_group.file.filename)
        elif hasattr(accessor, 'stochastic_h5_group') and accessor.stochastic_h5_group is not None:
            base_dir = os.path.dirname(accessor.stochastic_h5_group.file.filename)
            
        tsv_file = os.path.join(base_dir, "05_qa_features_tsv.jsonl")
        
        if os.path.exists(tsv_file):
            with open(tsv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        self.tsv_scores_cache[str(item["sample_id"])] = item["tsv_hallucination_score"]
            logger.info(f"✅ [TSV] 成功从 {base_dir} 加载 {len(self.tsv_scores_cache)} 条分数！")
        else:
            logger.warning(f"❌ [TSV] 找不到特征文件: {tsv_file}。请确认大管家已执行提取！")
            
        self.cache_loaded = True

    def predict_score(self, accessor) -> float:
        """
        向大管家输出幻觉概率。
        由于 AUROC 只看排序不看绝对数值，这里返回 e-05 级别的分数完全符合评测要求。
        """
        self._load_scores_if_needed(accessor)
        sid = str(accessor.sample_id)
        # 提取不到就给 NaN，防止污染其他正常跑完的数据
        return float(self.tsv_scores_cache.get(sid, float('nan')))