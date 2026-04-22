# data_utils/accessor.py
import h5py
import numpy as np
from typing import Dict, Any, List, Optional
import ml_dtypes  # 🚀 必须导入，为了解析 bfloat16
class SampleAccessor:
    """
    统一的数据访问器（Facade）。
    向不同类型的 Detector 屏蔽底层的存储细节，提供按需"拉取"数据的能力。

    支持的数据来源：
    1. metadata: 基础元数据 (prompt, model_output_text, eval_category 等)
    2. h5_group: 【原始生成】时的 hidden states (generated_tokens)
    3. qa_h5_file: Q+A 拼接后的 hidden states (CCS/PRISM/ICR Probe)
    4. stochastic_samples_dict: 【多次随机采样】的文本与概率 {sample_id: {"samples": [], "log_likelihoods": []}}
    5. stochastic_h5_group: 【多次随机采样】时的批量张量数据 (用于 EigenScore)
    """
    def __init__(
        self,
        sample_id: str,
        metadata: Dict[str, Any],
        h5_group: Optional[h5py.Group] = None,
        qa_h5_file: Optional[h5py.File] = None,
        method_type: Optional[str] = None,
        stochastic_samples_dict: Optional[Dict[str, Any]] = None,
        stochastic_h5_group: Optional[h5py.Group] = None  # 🎯 新增：多次采样的张量 Group
    ):
        self.sample_id = sample_id
        self.metadata = metadata
        self.h5_group = h5_group
        self.qa_h5_file = qa_h5_file
        self.method_type = method_type
        self.stochastic_samples_dict = stochastic_samples_dict
        self.stochastic_h5_group = stochastic_h5_group

    # ==========================================
    # 基础信息接口
    # ==========================================
    def get_prompt_text(self) -> str:
        return self.metadata.get("prompt", "")

    def get_model_output_text(self) -> str:
        return self.metadata.get("model_output_text", "")

    # ==========================================
    # 🎯 多次采样 (Stochastic) 专用接口
    # ==========================================
    def get_stochastic_samples(self) -> List[str]:
        """获取多次采样的文本列表"""
        data = self.stochastic_samples_dict.get(self.sample_id, {})
        if isinstance(data, dict):
            return data.get("samples", [])
        return data  # 兼容老版本 List[str] 格式

    def get_stochastic_logprobs(self) -> List[float]:
        """获取多次采样的序列级 Log-Likelihoods (用于 Semantic Entropy)"""
        data = self.stochastic_samples_dict.get(self.sample_id, {})
        if isinstance(data, dict):
            return data.get("log_likelihoods", [])
        return []

    def get_stochastic_hidden_states(self, sample_idx: int, layer_idx: int = -1) -> np.ndarray:
        """
        获取第 N 次随机采样在指定层的张量 (用于 EigenScore)
        
        Args:
            sample_idx: 第几次采样 (0-9)
            layer_idx: 层索引
        """
        if self.stochastic_h5_group is None:
            raise ValueError(f"Sample {self.sample_id}: 多次采样的 Hidden States 数据未挂载")
        
        # 路径结构: [sample_id]/stochastic_[i]/step_[j]/layer_[k]
        stochastic_run_key = f"stochastic_{sample_idx}"
        if stochastic_run_key not in self.stochastic_h5_group:
            raise KeyError(f"未找到第 {sample_idx} 次采样的张量数据")
            
        run_grp = self.stochastic_h5_group[stochastic_run_key]
        
        # 获取所有 step 键并排序
        step_keys = sorted(run_grp.keys(), key=lambda x: int(x.split('_')[1]))
        
        layer_tensors = []
        for step_key in step_keys:
            step_grp = run_grp[step_key]
            # 获取层名 (例如 layer_15)
            layer_keys = sorted(step_grp.keys(), key=lambda x: int(x.split('_')[1]))
            target_layer_key = layer_keys[layer_idx]
            
            data = step_grp[target_layer_key][:]
            if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
            layer_tensors.append(data)
            
        return np.stack(layer_tensors, axis=0) # [num_tokens, hidden_dim]

    # ==========================================
    # 原始生成 (Uncertainty) 接口
    # ==========================================
    def get_token_logprobs(self) -> List[float]:
        """
        提取原始回答（Main Output）中每个 Token 的对数概率。
        
        策略：
        1. 优先从 qa_h5_file (Step 2.2 补票数据) 中读取，因为补票通常包含最全的 logits。
        2. 如果没有，尝试从原始生成的 h5_group 中读取。
        """
        # --- 策略 1: 尝试从事后补票（QA 拼接推理）的文件中获取 ---
        if self.qa_h5_file and self.method_type:
            # 这里的 group_name 对应 runner.py 里的逻辑，如 "sample_001_prism"
            group_name = f"{self.sample_id}_{self.method_type}"
            if group_name in self.qa_h5_file:
                grp = self.qa_h5_file[group_name]
                if "logprobs" in grp:
                    # 返回的是整个 Answer 序列的对数概率列表
                    data = grp["logprobs"][:]
                    return data.tolist()

        # --- 策略 2: 尝试从原始生成（Step 2）的文件中获取 ---
        if self.h5_group and "logprobs" in self.h5_group:
            data = self.h5_group["logprobs"][:]
            # 处理 float16 可能存在的 void type 存储问题
            if data.dtype.kind == 'V':
                data = data.view(ml_dtypes.bfloat16).astype(np.float32)
            return data.tolist()

        # --- 兜底：返回空列表 ---
        return []

    # ==========================================
    # 原始生成 (Whitebox) 接口 - 逻辑保持并增强
    # ==========================================
    def get_hidden_states(self, layer_idx: int = -1, pooling: str = "mean") -> np.ndarray:
        if not self.h5_group or "generated_tokens" not in self.h5_group:
            raise ValueError(f"Sample {self.sample_id} 没有原始生成的隐藏状态数据")

        tokens_grp = self.h5_group["generated_tokens"]
        token_keys = sorted(tokens_grp.keys(), key=lambda x: int(x.split('_')[1]))
        
        # 预取第一步确定层信息
        first_step_layers = sorted(tokens_grp[token_keys[0]].keys(), key=lambda x: int(x.split('_')[1]))
        target_layer_key = first_step_layers[layer_idx]

        tensors = []
        for tk in token_keys:
            if target_layer_key in tokens_grp[tk]:
                data = tokens_grp[tk][target_layer_key][:]
                if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
                tensors.append(data)

        tensors_array = np.array(tensors)
        if pooling == "mean": return np.mean(tensors_array, axis=0)
        elif pooling == "last": return tensors_array[-1]
        # ... 其他 pooling 保持不变 ...
        return tensors_array[-1] # 默认 last

    # ==========================================
    # CCS/PRISM/ICR 接口 (Q+A 拼接数据)
    # ==========================================
    def get_qa_hidden_states(self, layer_idx: int = -1) -> np.ndarray:
        """获取 Q+A 拼接后的特征 (逻辑复用原版)"""
        if not self.qa_h5_file or not self.method_type:
            raise ValueError(f"Sample {self.sample_id}: QA 拼接数据未准备好")

        group_name = f"{self.sample_id}_{self.method_type}"
        grp = self.qa_h5_file[group_name]
        
        # 处理 icr_probe 的特殊层级
        layer_grp = grp["averaged"] if self.method_type == "icr_probe" else grp
        layer_keys = sorted([k for k in layer_grp.keys() if k.startswith("layer_")], key=lambda x: int(x.split('_')[1]))
        
        data = layer_grp[layer_keys[layer_idx]][:]
        if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
        return data

    def get_contrast_hidden_states(self, layer_idx: int = -1) -> tuple:
        """CCS 专用的正负对提取接口"""
        if not self.qa_h5_file or self.method_type != "ccs":
            raise ValueError(f"Sample {self.sample_id}: 缺少 CCS 特征")

        group_name = f"{self.sample_id}_ccs"
        grp = self.qa_h5_file[group_name]
        
        results = []
        for side in ["positive", "negative"]:
            sub_grp = grp[side]
            keys = sorted([k for k in sub_grp.keys() if k.startswith("layer_")], key=lambda x: int(x.split('_')[1]))
            data = sub_grp[keys[layer_idx]][:]
            if data.dtype.kind == 'V': data = data.view(np.float16)
            results.append(data)
            
        return results[0], results[1]