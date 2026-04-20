# /home/zfang1/Data/Lxy/Benchmark/data/generate_stochastic_samples.py
import os
import json
import time
import h5py
import torch
import ml_dtypes
import multiprocessing as mp
from typing import Dict, Any, List

# 🎯 纯正的邻居法则引入
from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder

# ==========================================
# 异步 I/O 专职子进程 (完美复刻你的原始设计)
# ==========================================
def stochastic_hdf5_writer_worker(data_queue, h5_filepath, jsonl_filepath, extract_hs):
    print(f"[Writer Process] 采样数据落盘进程已启动 (PID: {os.getpid()}).")
    # 如果不需要写入 HS，就不打开 H5，只写 JSONL
    h5_ctx = h5py.File(h5_filepath, 'a') if extract_hs else None
    
    with open(jsonl_filepath, 'a', encoding='utf-8') as f_json:
        while True:
            item = data_queue.get()
            if item is None: break  # 收到毒丸，安全退出

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
                                if layer_name in t_grp: del t_grp[layer_name]
                                t_grp.create_dataset(layer_name, data=tensor, compression="gzip")
                
                # 2. 写入 JSONL (追加了自评和采样的文本)
                f_json.write(json.dumps(item["meta_item"], ensure_ascii=False) + '\n')
                f_json.flush()
                if h5_ctx: h5_ctx.flush()
                
            except Exception as e:
                print(f"[Writer Process] Error saving {sample_id}: {e}")

    if h5_ctx: h5_ctx.close()
    print(f"[Writer Process] 所有 I/O 句柄已安全释放。")

# ==========================================
# 面向对象的真正继承 (OOP)
# ==========================================
class StochasticExtractor(HiddenStateExtractor):
    """
    继承自原生的 HiddenStateExtractor！
    复用多模态绕过加载逻辑、层级解析逻辑。
    专注于多次并发采样与实时自评推理，并封装了全流程。
    """
    
    def generate_and_extract_stochastic(self, prompt, layer_config, token_config, max_new_tokens, num_samples, generation_kwargs=None):
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        batch_results = []

        # =========================================================================
        # ⚠️ [保留代码] 原版高并发逻辑 (会导致 KV Cache 爆炸 OOM，已注释备用)
        # =========================================================================
        # with torch.no_grad():
        #     final_gen_kwargs = {
        #         **gen_kwargs,
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "max_new_tokens": max_new_tokens,
        #         "num_return_sequences": num_samples, # 🎯 并发生成 10 个
        #         "do_sample": True,                   # 🎯 强制开启采样
        #         "return_dict_in_generate": True,
        #         "output_hidden_states": True,
        #         "pad_token_id": self.tokenizer.pad_token_id
        #     }
        #     outputs = self.model.generate(**final_gen_kwargs)
        #
        #     transition_scores = None
        #     if hasattr(outputs, "scores"):
        #         transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        #
        # for i in range(num_samples):
        #     new_tokens = outputs.sequences[i, prompt_len:]
        #     valid_mask = (new_tokens != self.tokenizer.pad_token_id) & (new_tokens != self.tokenizer.eos_token_id)
        #     valid_tokens = new_tokens[valid_mask]
        #     total_generated = len(valid_tokens)
        #     full_output_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
        #
        #     token_logprobs = []
        #     if transition_scores is not None and total_generated > 0:
        #         token_logprobs = [round(float(s), 4) for s in transition_scores[i, :total_generated].cpu().numpy()]
        #
        #     target_indices = self._resolve_target_tokens(total_generated, token_config)
        #     filtered_tokens_data = []
        #
        #     if target_indices:
        #         layer_tensors = []
        #         for l in target_layers:
        #             hf_layer_idx = l + 1
        #             step_tensors = []
        #             for step in target_indices:
        #                 seq_idx = -1 if step == 0 else 0
        #                 step_tensors.append(outputs.hidden_states[step][hf_layer_idx][i, seq_idx, :])
        #             layer_tensors.append(torch.stack(step_tensors))
        #
        #         mega_tensor_gpu = torch.stack(layer_tensors)
        #         mega_tensor_cpu = mega_tensor_gpu.cpu().float().numpy().astype(ml_dtypes.bfloat16)
        #
        #         for idx_enum, step in enumerate(target_indices):
        #             token_id = valid_tokens[step].item()
        #             token_str = self.tokenizer.decode([token_id])
        #             step_states = {f"layer_{l:02d}": mega_tensor_cpu[j, idx_enum, :] for j, l in enumerate(target_layers)}
        #             filtered_tokens_data.append({
        #                 "token_id": token_id, "token_str": token_str,
        #                 "forward_idx": step, "backward_idx": step - total_generated,
        #                 "states": step_states
        #             })
        #
        #     batch_results.append({
        #         "text": full_output_text,
        #         "seq_logprob": round(sum(token_logprobs), 4) if token_logprobs else 0.0,
        #         "filtered_tokens_data": filtered_tokens_data
        #     })
        # =========================================================================

        # =========================================================================
        # 🎯 新版防 OOM 逻辑：串行生成，以时间换空间 (每次及时释放计算图)
        # =========================================================================
        for i in range(num_samples):
            with torch.no_grad():
                final_gen_kwargs = {
                    **gen_kwargs,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,         # 👈 强制每次只生成 1 条！
                    "do_sample": True,
                    "return_dict_in_generate": True,
                    "output_hidden_states": True,
                    "output_scores": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                outputs = self.model.generate(**final_gen_kwargs)

                transition_scores = None
                if hasattr(outputs, "scores"):
                    transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

            # 因为每次只生成 1 条，所以索引永远取 0
            new_tokens = outputs.sequences[0, prompt_len:]
            valid_mask = (new_tokens != self.tokenizer.pad_token_id) & (new_tokens != self.tokenizer.eos_token_id)
            valid_tokens = new_tokens[valid_mask]
            total_generated = len(valid_tokens)
            full_output_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)

            token_logprobs = []
            if transition_scores is not None and total_generated > 0:
                token_logprobs = [round(float(s), 4) for s in transition_scores[0, :total_generated].cpu().numpy()]

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
                # 完美继承：你极其优雅的 bfloat16 转换
                mega_tensor_cpu = mega_tensor_gpu.cpu().float().numpy().astype(ml_dtypes.bfloat16)

                for idx_enum, step in enumerate(target_indices):
                    token_id = valid_tokens[step].item()
                    token_str = self.tokenizer.decode([token_id])

                    step_states = {}
                    for j, l in enumerate(target_layers):
                        step_states[f"layer_{l:02d}"] = mega_tensor_cpu[j, idx_enum, :]

                    filtered_tokens_data.append({
                        "token_id": token_id, "token_str": token_str,
                        "forward_idx": step, "backward_idx": step - total_generated,
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

    def generate_single_response(self, prompt, max_new_tokens=100):
        """贪婪解码，供 Verbalize 和 Self-Eval 实时调用。内置了参数清理。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, 
                temperature=None, # 防警告清理
                top_p=None, 
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def process_stochastic_from_file(
            self,
            input_jsonl_path: str, output_h5_path: str, output_jsonl_path: str,
            layer_config: dict, token_config: dict, max_new_tokens: int, num_samples: int,
            system_prompt: str, num_shots: int, generation_kwargs: dict, template_kwargs: dict,
            run_verbalize: bool, run_self_evaluator: bool, extract_stochastic_hs: bool,
            max_queue_size: int = 10
    ):
        """高度内聚的主流程，和 process_from_file 一样优雅。"""
        os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
        
        # 1. 启动 I/O 进程
        data_queue = mp.Queue(maxsize=max_queue_size)
        writer_process = mp.Process(
            target=stochastic_hdf5_writer_worker,
            args=(data_queue, output_h5_path, output_jsonl_path, extract_stochastic_hs)
        )
        writer_process.daemon = True
        writer_process.start()

        # 2. 🎯 严谨继承：实例化 Builder (连 system_prompt 也传进去了)
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
                if line.strip(): dataset_items.append(json.loads(line))

        processed_count = 0
        try:
            for item in dataset_items:
                sample_id = item["sample_id"]
                print(f"[Main] 多次采样推理中: {sample_id}...")

                # 构建完全相同的 Prompt
                prompt_str = prompt_builder.build_prompt(target_item=item, few_shot_pool=dataset_items)
                
                # 获取 10 次串行生成结果
                batch_results = self.generate_and_extract_stochastic(
                    prompt_str, layer_config, token_config, max_new_tokens, num_samples, generation_kwargs
                )

                meta_item = item.copy()
                meta_item["stochastic_samples"] = [s["text"] for s in batch_results]
                meta_item["stochastic_log_likelihoods"] = [s["seq_logprob"] for s in batch_results]

                # 实时自评逻辑
                if run_verbalize:
                    v_p = f"Question: {prompt_str}\nAnswer: {item['model_output_text']}\nConfidence Score (0-1):"
                    meta_item["verbalize_response"] = self.generate_single_response(v_p, max_new_tokens=10)
                if run_self_evaluator:
                    se_p = f"Check consistency.\nQuestion: {prompt_str}\nAnswer: {item['model_output_text']}\nFinal Grade (Correct/Incorrect):"
                    meta_item["self_evaluator_raw"] = self.generate_single_response(se_p, max_new_tokens=150)

                # 扔进队列让子进程去慢慢存
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
            if writer_process.is_alive(): writer_process.terminate()
            print(f"[+] 采样提取流水线安全退出！处理了 {processed_count} 条。")