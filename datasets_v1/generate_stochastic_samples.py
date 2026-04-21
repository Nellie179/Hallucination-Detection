# /home/zfang1/Data/Lxy/Benchmark/data/generate_stochastic_samples.py
import os
import json
import time
import h5py
import torch
import math  # 🚨 必须导入，用于处理下溢的底层保护
import ml_dtypes
import multiprocessing as mp
from typing import Dict, Any, List

# 🎯 纯正的邻居法则引入
from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder

# ==========================================
# 异步 I/O 专职子进程
# ==========================================
def stochastic_hdf5_writer_worker(data_queue, h5_filepath, jsonl_filepath, extract_hs):
    print(f"[Writer Process] 采样数据落盘进程已启动 (PID: {os.getpid()}).")
    # 如果不需要写入 HS，就不打开 H5，只写 JSONL
    h5_ctx = h5py.File(h5_filepath, 'a') if extract_hs else None
    
    with open(jsonl_filepath, 'a', encoding='utf-8') as f_json:
        while True:
            item = data_queue.get()
            if item is None:
                break  # 收到毒丸，安全退出

            sample_id = item["sample_id"]
            try:
                # 1. 写入 H5 (如果开启了张量录制)
                if extract_hs and h5_ctx:
                    grp = h5_ctx.require_group(sample_id)
                    for i, sample_data in enumerate(item["batch_results"]):
                        run_grp = grp.require_group(f"stochastic_{i}")
                        for t_data in sample_data["filtered_tokens_data"]:
                            t_grp = run_grp.require_group(f"token_{t_data['forward_idx']:03d}")
                            t_grp.attrs["text"] = t_data["token_str"]
                            t_grp.attrs["forward_idx"] = t_data["forward_idx"]
                            t_grp.attrs["backward_idx"] = t_data["backward_idx"]
                            t_grp.attrs["token_id"] = t_data["token_id"]
                            for layer_name, tensor in t_data["states"].items():
                                if layer_name in t_grp:
                                    del t_grp[layer_name]
                                t_grp.create_dataset(layer_name, data=tensor, compression="gzip")
                
                # 2. 写入 JSONL
                f_json.write(json.dumps(item["meta_item"], ensure_ascii=False) + '\n')
                f_json.flush()
                if h5_ctx:
                    h5_ctx.flush()
                
            except Exception as e:
                print(f"[Writer Process] Error saving {sample_id}: {e}")

    if h5_ctx:
        h5_ctx.close()
    print(f"[Writer Process] 所有 I/O 句柄已安全释放。")

# ==========================================
# 面向对象的真正继承 (OOP)
# ==========================================
class StochasticExtractor(HiddenStateExtractor):
    """
    继承自原生的 HiddenStateExtractor！
    支持依赖注入模式：可直接接收外部传入的 model 实例，跳过重复加载。
    """
    
    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None):
        """
        🚀 [重构核心]: 依赖注入支持
        """
        if model is not None and tokenizer is not None:
            self.model_name = model_name or "injected_model"
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.model_kwargs = model_kwargs or {}
            
            # ==========================================
            # 🚨 [热修复补丁]: 补上缺失的模型层数推断
            # ==========================================
            if hasattr(self.model.config, "num_hidden_layers"):
                self.total_layers = self.model.config.num_hidden_layers
            else:
                self.total_layers = self.model.config.text_config.num_hidden_layers

            print(f"[*] StochasticExtractor 成功接收指挥官注入的模型实例 (设备: {self.device})")
        else:
            # 兼容老版本，如果没有注入，则调用父类去加载
            super().__init__(model_name, model_kwargs)

    def generate_and_extract_stochastic(self, prompt, layer_config, token_config, max_new_tokens, num_samples, generation_kwargs=None):
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        batch_results = []

        # =========================================================================
        # 🎯 严谨对齐原论文：Multinomial Beam Sampling (支持 num_beams > 1)
        # =========================================================================
        for i in range(num_samples):
            with torch.no_grad():
                final_gen_kwargs = {
                    **gen_kwargs,                        # 接受外部传入的 num_beams=5 和 do_sample=True
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,           # 强制每次只生成 1 条以防止 KV Cache OOM！
                    "return_dict_in_generate": True,
                    "output_hidden_states": True,
                    "output_scores": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                outputs = self.model.generate(**final_gen_kwargs)

                transition_scores = None
                if hasattr(outputs, "scores"):
                    # 🚨 核心修复：一旦开启 Beam Search，必须传入 beam_indices 否则概率会全部下溢为 -inf！
                    if hasattr(outputs, "beam_indices") and outputs.beam_indices is not None:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences, outputs.scores, beam_indices=outputs.beam_indices, normalize_logits=True
                        )
                    else:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )

            # 因为每次只生成 1 条，所以索引永远取 0
            new_tokens = outputs.sequences[0, prompt_len:]
            valid_mask = (new_tokens != self.tokenizer.pad_token_id) & (new_tokens != self.tokenizer.eos_token_id)
            valid_tokens = new_tokens[valid_mask]
            total_generated = len(valid_tokens)
            full_output_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)

            token_logprobs = []
            if transition_scores is not None and total_generated > 0:
                # 🚨 终极防御：提取概率时进行安全钳制
                for s in transition_scores[0, :total_generated].cpu().numpy():
                    val = float(s)
                    if math.isinf(val) or math.isnan(val):
                        val = -15.0  # 提供一个极小的合理概率兜底
                    token_logprobs.append(round(val, 4))

            target_indices = self._resolve_target_tokens(total_generated, token_config)
            filtered_tokens_data = []

            if target_indices:
                layer_tensors = []
                for l in target_layers:
                    hf_layer_idx = l + 1
                    step_tensors = []
                    for step in target_indices:
                        seq_idx = -1 if step == 0 else 0
                        # 🎯 索引固定取 0
                        step_tensors.append(outputs.hidden_states[step][hf_layer_idx][0, seq_idx, :])
                    layer_tensors.append(torch.stack(step_tensors))

                mega_tensor_gpu = torch.stack(layer_tensors)
                mega_tensor_cpu = mega_tensor_gpu.cpu().float().numpy().astype(ml_dtypes.bfloat16)

                for idx_enum, step in enumerate(target_indices):
                    token_id = valid_tokens[step].item()
                    token_str = self.tokenizer.decode([token_id])

                    step_states = {}
                    for j, l in enumerate(target_layers):
                        step_states[f"layer_{l:02d}"] = mega_tensor_cpu[j, idx_enum, :]

                    filtered_tokens_data.append({
                        "token_id": token_id,
                        "token_str": token_str,
                        "forward_idx": step,
                        "backward_idx": step - total_generated,
                        "states": step_states
                    })

            batch_results.append({
                "text": full_output_text,
                "seq_logprob": round(sum(token_logprobs), 4) if token_logprobs else 0.0,
                "filtered_tokens_data": filtered_tokens_data
            })
            
            # 🎯 致命一击：每次循环结束，强制释放本次产生的庞大计算图和 KV Cache！
            del outputs, transition_scores
            torch.cuda.empty_cache()

        return batch_results

    def process_stochastic_from_file(
            self,
            input_jsonl_path: str, output_h5_path: str, output_jsonl_path: str,
            layer_config: dict, token_config: dict, max_new_tokens: int, num_samples: int,
            system_prompt: str, num_shots: int, generation_kwargs: dict, template_kwargs: dict,
            run_verbalize: bool, run_self_evaluator: bool, extract_stochastic_hs: bool,
            max_queue_size: int = 10
    ):
        os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
        
        # 1. 启动 I/O 进程
        data_queue = mp.Queue(maxsize=max_queue_size)
        writer_process = mp.Process(
            target=stochastic_hdf5_writer_worker,
            args=(data_queue, output_h5_path, output_jsonl_path, extract_stochastic_hs)
        )
        writer_process.daemon = True
        writer_process.start()

        print("[*] 正在加载数据池并初始化 Prompt 渲染引擎...")
        prompt_builder = LLMPromptBuilder(
            model_name=self.model_name,
            global_system_prompt=system_prompt,
            num_shots=num_shots,
            template_kwargs=template_kwargs
        )

        dataset_items = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset_items.append(json.loads(line))

        processed_count = 0
        try:
            for item in dataset_items:
                sample_id = item["sample_id"]
                print(f"[Main] 多次采样推理中: {sample_id}...")

                prompt_str = prompt_builder.build_prompt(
                    target_item=item,
                    few_shot_pool=dataset_items
                )
                
                batch_results = self.generate_and_extract_stochastic(
                    prompt_str, layer_config, token_config, max_new_tokens, num_samples, generation_kwargs
                )

                meta_item = item.copy()
                meta_item["stochastic_samples"] = [s["text"] for s in batch_results]
                meta_item["stochastic_log_likelihoods"] = [s["seq_logprob"] for s in batch_results]

                data_queue.put({
                    "sample_id": sample_id,
                    "batch_results": batch_results,
                    "meta_item": meta_item
                }, block=True)
                processed_count += 1

        except KeyboardInterrupt:
            print("\n[!] 🛑 接收到打断信号，准备安全退出...")
        finally:
            data_queue.put(None)
            writer_process.join(timeout=10)
            if writer_process.is_alive():
                writer_process.terminate()
            print(f"[+] 采样提取流水线安全退出！处理了 {processed_count} 条。")