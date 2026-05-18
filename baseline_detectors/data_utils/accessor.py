import h5py
import numpy as np
from typing import Dict, Any, List, Optional
import ml_dtypes


class SampleAccessor:
    def __init__(
            self,
            sample_id: str,
            metadata: Dict[str, Any],
            h5_group: Optional[h5py.Group] = None,
            qa_h5_file: Optional[h5py.File] = None,
            method_type: Optional[str] = None,
            stochastic_samples_dict: Optional[Dict[str, Any]] = None,
            stochastic_h5_group: Optional[h5py.Group] = None
    ):
        self.sample_id = sample_id
        self.metadata = metadata
        self.h5_group = h5_group
        self.qa_h5_file = qa_h5_file
        self.method_type = method_type
        self.stochastic_samples_dict = stochastic_samples_dict
        self.stochastic_h5_group = stochastic_h5_group

    def get_prompt_text(self) -> str:
        return self.metadata.get("prompt", "")

    def get_model_output_text(self) -> str:
        return self.metadata.get("model_output_text", "")

    def get_stochastic_samples(self) -> List[str]:
        data = self.stochastic_samples_dict.get(self.sample_id, {})
        if isinstance(data, dict):
            return data.get("samples", [])
        return data

    def get_stochastic_logprobs(self) -> List[float]:
        data = self.stochastic_samples_dict.get(self.sample_id, {})
        if isinstance(data, dict):
            return data.get("log_likelihoods", [])
        return []

    def get_stochastic_hidden_states(self, sample_idx: int, layer_idx: int = -1) -> np.ndarray:
        if self.stochastic_h5_group is None:
            raise ValueError(f"Sample {self.sample_id}: 多次采样的 Hidden States 数据未挂载")

        stochastic_run_key = f"stochastic_{sample_idx}"
        if stochastic_run_key not in self.stochastic_h5_group:
            raise KeyError(f"未找到第 {sample_idx} 次采样的张量数据")

        run_grp = self.stochastic_h5_group[stochastic_run_key]

        step_keys = sorted(run_grp.keys(), key=lambda x: int(x.split('_')[1]))

        layer_tensors = []
        for step_key in step_keys:
            step_grp = run_grp[step_key]
            layer_keys = sorted(step_grp.keys(), key=lambda x: int(x.split('_')[1]))
            target_layer_key = layer_keys[layer_idx]

            data = step_grp[target_layer_key][:]
            if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
            layer_tensors.append(data)

        return np.stack(layer_tensors, axis=0)

    def get_token_logprobs(self) -> List[float]:
        if self.qa_h5_file and self.method_type:
            group_name = f"{self.sample_id}_{self.method_type}"
            if group_name in self.qa_h5_file:
                grp = self.qa_h5_file[group_name]
                if "logprobs" in grp:
                    data = grp["logprobs"][:]
                    return data.tolist()

        if self.h5_group and "logprobs" in self.h5_group:
            data = self.h5_group["logprobs"][:]
            if data.dtype.kind == 'V':
                data = data.view(ml_dtypes.bfloat16).astype(np.float32)
            return data.tolist()

        return []

    def get_hidden_states(self, layer_idx: int = -1, pooling: str = "mean") -> np.ndarray:
        if not self.h5_group or "generated_tokens" not in self.h5_group:
            raise ValueError(f"Sample {self.sample_id} 没有原始生成的隐藏状态数据")

        tokens_grp = self.h5_group["generated_tokens"]
        token_keys = sorted(tokens_grp.keys(), key=lambda x: int(x.split('_')[1]))

        first_step_layers = sorted(tokens_grp[token_keys[0]].keys(), key=lambda x: int(x.split('_')[1]))
        target_layer_key = first_step_layers[layer_idx]

        tensors = []
        for tk in token_keys:
            if target_layer_key in tokens_grp[tk]:
                data = tokens_grp[tk][target_layer_key][:]
                if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
                tensors.append(data)

        tensors_array = np.array(tensors)
        if pooling == "mean":
            return np.mean(tensors_array, axis=0)
        elif pooling == "last":
            return tensors_array[-1]
        return tensors_array[-1]

    def get_qa_hidden_states(self, layer_idx: int = -1) -> np.ndarray:
        if not self.qa_h5_file or not self.method_type:
            raise ValueError(f"Sample {self.sample_id}: QA 拼接数据未准备好")

        group_name = f"{self.sample_id}_{self.method_type}"
        grp = self.qa_h5_file[group_name]

        layer_grp = grp["averaged"] if self.method_type == "icr_probe" else grp
        layer_keys = sorted([k for k in layer_grp.keys() if k.startswith("layer_")], key=lambda x: int(x.split('_')[1]))

        data = layer_grp[layer_keys[layer_idx]][:]
        if data.dtype.kind == 'V': data = data.view(ml_dtypes.bfloat16).astype(np.float32)
        return data

    def get_contrast_hidden_states(self, layer_idx: int = -1) -> tuple:
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