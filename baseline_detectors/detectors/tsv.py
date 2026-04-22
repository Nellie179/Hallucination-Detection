import json
import logging
import os
from detectors.base import BaseDetector
from detectors.registry import register_detector

logger = logging.getLogger(__name__)

@register_detector("tsv")
class TSVDetector(BaseDetector):
    """
    TSV (Task-Steering Vector) 幻觉探测器。
    [设计理念]: 极轻量级读取器。底层特征和分数已在 extract_tsv_features.py 中计算完毕，
    此处仅负责高效缓存读取并对接到 Universal Schema 评测管线。
    """
    def __init__(self, name="tsv", **kwargs):
        super().__init__(name, **kwargs)
        self.tsv_scores_cache = {}
        self.cache_loaded = False

    def _load_scores_if_needed(self, accessor):
        # 如果已经加载过缓存，直接跳过，保证 O(1) 的查询速度
        if self.cache_loaded:
            return

        # 动态定位到当前实验的输出目录 (依托于 accessor 的上下文)
        base_dir = os.path.dirname(accessor.h5_path) if hasattr(accessor, 'h5_path') and accessor.h5_path else "."
        tsv_file = os.path.join(base_dir, "05_qa_features_tsv.jsonl")
        
        if not os.path.exists(tsv_file):
            logger.warning(f"❌ [TSV] 找不到特征文件: {tsv_file}。请确认大管家已执行 TSVFeatureExtractor！")
            self.cache_loaded = True # 标记为已尝试加载，避免后续万条数据疯狂刷屏报错
            return
            
        # 一次性将所有分数读入内存字典
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.tsv_scores_cache[str(item["sample_id"])] = item["tsv_hallucination_score"]
                    
        self.cache_loaded = True
        logger.info(f"✅ [TSV] 成功将 {len(self.tsv_scores_cache)} 条预计算的 TSV 分数加载至内存！")

    def predict_score(self, accessor) -> float:
        """
        核心打分接口。
        返回的值越大，代表该样本是幻觉的概率越高。
        """
        self._load_scores_if_needed(accessor)
        
        sid = str(accessor.sample_id)
        
        if sid in self.tsv_scores_cache:
            # extract_tsv_features.py 中存入的已经是“幻觉类的 Softmax 概率”
            # 直接返回该值，完美对齐 AUROC 的单调性要求
            return float(self.tsv_scores_cache[sid])
        else:
            # 如果某条数据因意外丢失，返回 NaN 避免污染整体榜单
            logger.debug(f"[TSV] 样本 {sid} 缺失 TSV 分数。")
            return float('nan')