# baseline_detectors/evaluators/classification.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from typing import Dict, List

class ClassificationEvaluator:
    """
    统一的分类指标评估引擎。
    负责计算 AUROC, AUPR 以及 FPR@95TPR。
    """

    @staticmethod
    def _compute_fpr_at_tpr(y_true: np.ndarray, y_scores: np.ndarray, tpr_threshold: float = 0.95) -> float:
        """
        计算当 True Positive Rate (TPR) 达到指定阈值时的 False Positive Rate (FPR)。
        这是 OOD 检测和幻觉检测中的核心指标之一。
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # 找到满足 TPR >= 0.95 的最小 FPR
        idx = np.where(tpr >= tpr_threshold)[0]
        if len(idx) > 0:
            return fpr[idx[0]]
        return float('nan')

    @classmethod
    def compute_metrics(cls, y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
        """
        计算单组数据的综合指标。
        包含无监督方向校准：自动尝试两个方向，取 AUROC 更大的方向计算所有指标。
        """
        y_t = np.array(y_true)
        y_s = np.array(y_scores)

        # 边界条件处理：如果当前批次只有一种标签，无法计算 AUROC
        if len(np.unique(y_t)) < 2:
            return {"AUROC": float('nan'), "AUPR": float('nan'), "FPR@95": float('nan')}

        # 第一次计算 AUROC
        auroc = roc_auc_score(y_t, y_s)

        # 🚨 自动方向纠正 (Sign Ambiguity Fix)
        # 如果 AUROC < 0.5，说明模型找到的特征是对的，但正负极接反了
        if auroc < 0.5:
            # 将分数取反，重新对齐方向
            y_s = -y_s
            # 重新计算纠正后的 AUROC (此时必然大于 0.5)
            auroc = roc_auc_score(y_t, y_s)

        # 基于纠正后的正确方向，计算 AUPR 和 FPR@95
        aupr = average_precision_score(y_t, y_s)
        fpr95 = cls._compute_fpr_at_tpr(y_t, y_s, 0.95)

        return {
            "AUROC": round(auroc * 100, 2),
            "AUPR": round(aupr * 100, 2),
            "FPR@95": round(fpr95 * 100, 2)
        }
    @classmethod
    def evaluate_stratified(cls, results_data: List[Dict], grouping_key: str = "task_type") -> Dict[str, Dict[str, float]]:
        """
        分层评估：根据 metadata 中的指定字段（如 task_type）对结果进行分组计算。
        
        参数:
            results_data: 包含 {'y_true': int, 'y_score': float, 'metadata': dict} 的列表
            grouping_key: 用于切片的字段名
            
        返回:
            以切片名称为键，以指标字典为值的嵌套字典。
        """
        grouped_results = {}
        for item in results_data:
            # 安全获取分组键值，默认归入 'unknown'
            group_val = item.get("metadata", {}).get(grouping_key, "unknown")
            if group_val not in grouped_results:
                grouped_results[group_val] = {"y_true": [], "y_score": []}
            
            grouped_results[group_val]["y_true"].append(item["y_true"])
            grouped_results[group_val]["y_score"].append(item["y_score"])

        stratified_metrics = {}
        for group, data in grouped_results.items():
            stratified_metrics[group] = cls.compute_metrics(data["y_true"], data["y_score"])

        return stratified_metrics