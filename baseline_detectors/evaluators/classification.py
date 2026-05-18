import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from typing import Dict, List


class ClassificationEvaluator:

    @staticmethod
    def _compute_fpr_at_tpr(y_true: np.ndarray, y_scores: np.ndarray, tpr_threshold: float = 0.95) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.where(tpr >= tpr_threshold)[0]
        if len(idx) > 0:
            return fpr[idx[0]]
        return float('nan')

    @classmethod
    def compute_metrics(cls, y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
        y_t = np.array(y_true)
        y_s = np.array(y_scores)

        if len(np.unique(y_t)) < 2:
            return {"AUROC": float('nan'), "AUPR": float('nan'), "FPR@95": float('nan')}

        auroc = roc_auc_score(y_t, y_s)

        if auroc < 0.5:
            y_s = -y_s
            auroc = roc_auc_score(y_t, y_s)

        aupr = average_precision_score(y_t, y_s)
        fpr95 = cls._compute_fpr_at_tpr(y_t, y_s, 0.95)

        return {
            "AUROC": round(auroc * 100, 2),
            "AUPR": round(aupr * 100, 2),
            "FPR@95": round(fpr95 * 100, 2)
        }

    @classmethod
    def evaluate_stratified(cls, results_data: List[Dict], grouping_key: str = "task_type") -> Dict[
        str, Dict[str, float]]:
        grouped_results = {}
        for item in results_data:
            group_val = item.get("metadata", {}).get(grouping_key, "unknown")
            if group_val not in grouped_results:
                grouped_results[group_val] = {"y_true": [], "y_score": []}

            grouped_results[group_val]["y_true"].append(item["y_true"])
            grouped_results[group_val]["y_score"].append(item["y_score"])

        stratified_metrics = {}
        for group, data in grouped_results.items():
            stratified_metrics[group] = cls.compute_metrics(data["y_true"], data["y_score"])

        return stratified_metrics