# baseline_detectors/detectors/sep.py
"""
SEP (Semantic Entropy Probes) Detector - 语义熵线性探针检测器

原理：
    传统的semantic entropy需要采样多个响应(5-10倍计算成本),
    SEP通过训练线性探针直接从单次生成的hidden states预测semantic entropy。

方法：
    1. 从hidden states的关键token位置提取特征
       - TBG (Token Before Generation): 生成开始前的token
       - SLT (Selected Last Token): 最后一个token

    2. 训练线性探针预测semantic entropy
       - 使用少量标注数据训练
       - 线性模型,参数少,泛化好

    3. 测试时只需单次forward pass,无需采样

优势：
    - 计算成本几乎为零(相比semantic entropy)
    - 泛化能力强,OOD性能好
    - 简单高效,易于部署

参考文献：
    Kossen et al. "Semantic Entropy Probes: Robust and Cheap Hallucination
    Detection in LLMs"
    ICML 2024
    https://arxiv.org/abs/2406.15927
    https://github.com/OATML/semantic-entropy-probes

依赖：
    numpy, sklearn
"""

# baseline_detectors/detectors/sep.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from detectors.base import BaseDetector
from detectors.registry import register_detector

@register_detector("sep")
class SEPDetector(BaseDetector):
    """
    Semantic Entropy Probes (SEP) - ICML 2024 (Official Logic Implementation)
    依据 OATML 官方实现：使用线性探针从特定 Token 位置预测幻觉。
    """
    def __init__(self, name="sep", pos_type="both", layer_idx=-1, **kwargs):
        """
        Args:
            pos_type: 'tbg' (生成前), 'slt' (回答末尾), 或 'both' (二者拼接)
            layer_idx: 特征层索引，默认 -1 (最后一层)
        """
        super().__init__(name, **kwargs)
        self.requires_qa_features = True
        self.pos_type = pos_type
        self.layer_idx = layer_idx
        
        # 官方实现核心：L2 正则化 Logistic Regression
        self.clf = LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            solver='lbfgs', 
            class_weight='balanced' # 针对幻觉不平衡样本的优化
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_sep_feature(self, accessor):
        """严格按照 extract_qa_hidden_states.py 存入的 sep_points 路径读取"""
        sid = accessor.sample_id
        method_key = f"{sid}_sep"
        
        if method_key not in accessor.qa_h5_file:
            raise KeyError(f"❌ 样本 {sid} 缺少 SEP 特征。请先运行提取脚本并在方法中指定 'sep'。")
        
        grp = accessor.qa_h5_file[method_key]["sep_points"]
        
        # 如果是 -1，动态探测最后一层
        l_idx = self.layer_idx
        if l_idx == -1:
            l_idx = max([int(k.split('_')[-1]) for k in grp.keys() if "tbg_layer_" in k])

        # 提取 TBG 和 SLT (这是论文的核心创新点)
        tbg = grp[f"tbg_layer_{l_idx}"][:]
        slt = grp[f"slt_layer_{l_idx}"][:]

        if self.pos_type == "tbg":
            return tbg
        elif self.pos_type == "slt":
            return slt
        else:
            # 论文中最强的模式：[TBG, SLT] 拼接
            return np.concatenate([tbg, slt], axis=-1)

    def fit(self, train_accessors):
        """SEP 是监督学习探测器，必须在训练集上拟合"""
        X_train, y_train = [], []
        
        for acc in train_accessors:
            try:
                feat = self._get_sep_feature(acc)
                # 标签映射：hallucination -> 1, correct -> 0
                label = 1 if acc.metadata.get("eval_category") == "hallucination" else 0
                X_train.append(feat)
                y_train.append(label)
            except Exception:
                continue
        
        if len(set(y_train)) < 2:
            print(f"[!] {self.name} 训练终止：样本类别不足或特征读取失败。")
            return

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 官方 Standardizer 流程
        X_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_scaled, y_train)
        self.is_fitted = True
        print(f"[+] {self.name} 线性探针训练完成 (样本数: {len(X_train)})")

    def predict_score(self, accessor):
        """输出幻觉概率分数"""
        if not self.is_fitted:
            return 0.5
        
        try:
            feat = self._get_sep_feature(accessor).reshape(1, -1)
            X_test_scaled = self.scaler.transform(feat)
            
            # predict_proba 返回 [P(correct), P(hallucination)]
            # 我们需要幻觉的概率作为 Score
            probs = self.clf.predict_proba(X_test_scaled)[0]
            return float(probs[1])
        except Exception:
            return 0.5