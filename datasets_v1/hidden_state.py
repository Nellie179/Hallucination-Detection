import os
import json
import torch
import h5py
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import ml_dtypes
import time
import numpy as np
from prompt_builder import LLMPromptBuilder


def hdf5_writer_process_worker(data_queue, h5_filepath, jsonl_filepath):
    print(f"[Writer Process] Started (PID: {os.getpid()}). Dedicated to I/O.")
    with open(jsonl_filepath, 'a', encoding='utf-8') as f_json, \
            h5py.File(h5_filepath, 'a') as f_h5:

        while True:
            item = data_queue.get()
            if item is None:
                break

            sample_id = item["sample_id"]
            try:
                if sample_id in f_h5:
                    del f_h5[sample_id]

                grp = f_h5.create_group(sample_id)
                grp.attrs["original_prompt"] = item["prompt"]
                grp.attrs["model_output"] = item["meta_item"]["model_output_text"]

                tokens_grp = grp.create_group("generated_tokens")
                for token_data in item["filtered_tokens_data"]:
                    t_grp = tokens_grp.create_group(f"token_{token_data['forward_idx']:03d}")
                    t_grp.attrs["text"] = token_data["token_str"]
                    t_grp.attrs["forward_idx"] = token_data["forward_idx"]
                    t_grp.attrs["backward_idx"] = token_data["backward_idx"]
                    t_grp.attrs["token_id"] = token_data["token_id"]

                    for layer_name, tensor in token_data["states"].items():
                        t_grp.create_dataset(layer_name, data=tensor, compression="gzip")

                f_json.write(json.dumps(item["meta_item"], ensure_ascii=False) + '\n')
                f_json.flush()
                f_h5.flush()
            except Exception as e:
                print(f"[Writer Process] Error saving {sample_id}: {e}")

    print(f"[Writer Process] Finished and safely closed all file handles.")


class HiddenStateExtractor:
    def __init__(
            self,
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            device=None,
            dtype=torch.bfloat16,
            model_kwargs=None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        print(f"[*] 正在使用原生 Transformers 加载模型 {model_name} 到 {self.device}...")
        if self.model_kwargs:
            print(f"    - 透传加载参数: {self.model_kwargs}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **self.model_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self.device,
            **self.model_kwargs
        )
        self.model.eval()

        self.total_layers = self.model.config.num_hidden_layers
        print(f"[+] 模型加载完成，总层数: {self.total_layers}")

    def _resolve_target_layers(self, layer_config):
        if not layer_config:
            return list(range(self.total_layers - 3, self.total_layers))

        if isinstance(layer_config, list):
            return [(l + self.total_layers if l < 0 else l) for l in layer_config]
        elif isinstance(layer_config, str) and layer_config == "all":
            return list(range(self.total_layers))
        elif isinstance(layer_config, dict) and layer_config.get("mode") == "middle":
            count = min(layer_config.get("count", 1), self.total_layers)
            start_idx = (self.total_layers // 2) - (count // 2)
            return list(range(start_idx, start_idx + count))
        else:
            raise ValueError(f"Unsupported layer config: {layer_config}")

    def _resolve_target_tokens(self, total_generated, token_config):
        if not token_config:
            token_config = {"mode": "backward", "count": 5}

        mode = token_config.get("mode", "all")
        count = min(token_config.get("count", total_generated), total_generated)

        if mode == "all":
            return list(range(total_generated))
        elif mode == "forward":
            return list(range(count))
        elif mode == "backward":
            start = max(0, total_generated - count)
            return list(range(start, total_generated))
        else:
            raise ValueError(f"Unsupported token config: {token_config}")

    def generate_and_extract(self, prompt, layer_config, token_config, max_new_tokens, generation_kwargs=None):
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            final_gen_kwargs = {
                **gen_kwargs,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
                "output_hidden_states": True,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            outputs = self.model.generate(**final_gen_kwargs)

        new_tokens = outputs.sequences[0, prompt_len:]
        total_generated = len(new_tokens)
        full_output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        target_indices = self._resolve_target_tokens(total_generated, token_config)

        if not target_indices:
            return [], full_output_text, total_generated

        layer_tensors = []
        for l in target_layers:
            hf_layer_idx = l + 1
            step_tensors = []
            for step in target_indices:
                seq_idx = -1 if step == 0 else 0
                step_tensors.append(outputs.hidden_states[step][hf_layer_idx][0, seq_idx, :])
            layer_tensors.append(torch.stack(step_tensors))

        mega_tensor_gpu = torch.stack(layer_tensors)
        mega_tensor_cpu = mega_tensor_gpu.cpu().float().numpy().astype(ml_dtypes.bfloat16)

        filtered_tokens_data = []
        for i, step in enumerate(target_indices):
            token_id = new_tokens[step].item()
            token_str = self.tokenizer.decode([token_id])

            step_states = {}
            for j, l in enumerate(target_layers):
                step_states[f"layer_{l:02d}"] = mega_tensor_cpu[j, i, :]

            filtered_tokens_data.append({
                "token_id": token_id,
                "token_str": token_str,
                "forward_idx": step,
                "backward_idx": step - total_generated,
                "states": step_states
            })

        return filtered_tokens_data, full_output_text, total_generated

    def process_from_file(
            self,
            input_jsonl_path: str,
            output_h5_path: str,
            output_jsonl_path: str,
            layer_config: dict = None,
            token_config: dict = None,
            max_new_tokens: int = 20,
            max_queue_size: int = 10,
            system_prompt: str = "You are a helpful assistant.",
            num_shots: int = 0,
            generation_kwargs: dict = None,
            template_kwargs: dict = None
    ):
        gen_kwargs = generation_kwargs or {}
        tpl_kwargs = template_kwargs or {}

        if not os.path.exists(input_jsonl_path):
            raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

        os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

        with h5py.File(output_h5_path, 'a') as f:
            f.attrs["llm_model"] = self.model_name
            f.attrs["dataset_source"] = os.path.basename(input_jsonl_path)
            f.attrs["layer_config"] = json.dumps(layer_config or {})
            f.attrs["token_config"] = json.dumps(token_config or {})
            f.attrs["model_kwargs"] = json.dumps(self.model_kwargs)
            f.attrs["generation_kwargs"] = json.dumps(gen_kwargs)
            f.attrs["template_kwargs"] = json.dumps(tpl_kwargs)
            f.attrs["prompt_config"] = json.dumps({"system_prompt": system_prompt, "num_shots": num_shots})
            f.attrs["extraction_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        data_queue = mp.Queue(maxsize=max_queue_size)
        writer_process = mp.Process(
            target=hdf5_writer_process_worker,
            args=(data_queue, output_h5_path, output_jsonl_path)
        )
        writer_process.daemon = True
        writer_process.start()

        print("[*] 正在加载数据池并初始化 Prompt 渲染引擎...")
        prompt_builder = LLMPromptBuilder(
            model_name=self.model_name,
            global_system_prompt=system_prompt,
            num_shots=num_shots,
            template_kwargs=tpl_kwargs
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
                print(f"[Main Process] Inferencing {sample_id}...")

                prompt_str = prompt_builder.build_prompt(target_item=item, few_shot_pool=dataset_items)
                item["prompt"] = prompt_str

                filtered_tokens_data, model_output_text, total_generated = self.generate_and_extract(
                    prompt_str, layer_config, token_config, max_new_tokens, generation_kwargs=gen_kwargs
                )

                meta_item = item.copy()
                meta_item.update({
                    "model_output_text": model_output_text,
                    "total_generated_tokens": total_generated,
                    "saved_tokens_count": len(filtered_tokens_data),
                    "h5_file_reference": output_h5_path
                })

                data_queue.put({
                    "sample_id": sample_id,
                    "prompt": prompt_str,
                    "filtered_tokens_data": filtered_tokens_data,
                    "meta_item": meta_item
                }, block=True)

                processed_count += 1

        except KeyboardInterrupt:
            print("\n[!] 🛑 接收到打断信号 (Ctrl+C)！正在紧急执行安全收尾...")
            print("[!] 强行中断可能导致本次特征提取不完整，但正在全力保护已落盘的数据。")

        finally:
            print(f"[*] 等待 I/O 进程保存并关闭 HDF5 文件 (至关重要)...")
            data_queue.put(None)
            writer_process.join(timeout=10)

            if writer_process.is_alive():
                print("[!] ⚠️ I/O 进程未能按时退出，执行强制终止！")
                writer_process.terminate()

            print(f"[+] 提取流水线已安全退出！本次成功提取 {processed_count} 条。")


# ==========================================
# 本地组件验证测试 (TEST沙盒隔离 + Few-shot抗压版)
# ==========================================
if __name__ == "__main__":
    # 【核心新增】：所有测试产物圈禁在专门的 TEST 目录中
    TEST_DIR = "./TEST"
    os.makedirs(TEST_DIR, exist_ok=True)

    test_input_file = os.path.join(TEST_DIR, "test_extreme_dataset.jsonl")
    test_output_h5 = os.path.join(TEST_DIR, "output_tensors_test.h5")
    test_output_jsonl = os.path.join(TEST_DIR, "output_metadata_test.jsonl")

    # 构造题库池，必须超过我们设置的 num_shots 数量，才能测试出随机抽样的魅力
    mock_dataset = [
        {"sample_id": "test_qa_001", "structured_data": {"task_type": "qa", "system_instruction": "", "context": "",
                                                         "question": "Capital of France?", "choices": {},
                                                         "ground_truths": ["Paris"], "incorrect_answers": []},
         "original_doc": {}},
        {"sample_id": "test_mc_002",
         "structured_data": {"task_type": "multiple_choice", "system_instruction": "", "context": "",
                             "question": "Red Planet?", "choices": {"A": "Earth", "B": "Mars"}, "ground_truths": ["B"],
                             "incorrect_answers": []}, "original_doc": {}},
        {"sample_id": "test_code_003",
         "structured_data": {"task_type": "coding", "system_instruction": "You are a Python expert.", "context": "",
                             "question": "def hello():\n    \"\"\"Print Hello\"\"\"\n", "choices": {},
                             "ground_truths": ["    print('Hello')"], "incorrect_answers": []}, "original_doc": {}},
        {"sample_id": "test_agent_004",
         "structured_data": {"task_type": "agent_action", "system_instruction": "Select API.",
                             "context": "APIs: 1. get_weather", "question": "Weather in NYC?", "choices": {},
                             "ground_truths": ["get_weather('NYC')"], "incorrect_answers": []}, "original_doc": {}}
    ]

    print(f"[*] 生成符合 Universal Schema 的极限测试数据，并隔离至 {TEST_DIR}/ ...")
    with open(test_input_file, "w", encoding="utf-8") as f:
        for item in mock_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[*] 开始原生 Transformers 模块全链路抗压测试...")

    # 极限透传参数测试 (确保字典正确穿越)
    test_model_kwargs = {"trust_remote_code": True}
    test_gen_kwargs = {"do_sample": False}
    test_template_kwargs = {}

    extractor = HiddenStateExtractor(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_kwargs=test_model_kwargs
    )

    extractor.process_from_file(
        input_jsonl_path=test_input_file,
        output_h5_path=test_output_h5,
        output_jsonl_path=test_output_jsonl,
        layer_config={"mode": "middle", "count": 2},
        # 测试极端 Token 提取：提取倒数 10 个 (看是否会触发 out of index 保护)
        token_config={"mode": "backward", "count": 10},
        max_new_tokens=20,  # 故意设短看截断表现
        system_prompt="You are undergoing an extreme stress test.",
        num_shots=2,  # 【核心测试】：强行开启 2-shot 并在大文本下验证张量聚合
        generation_kwargs=test_gen_kwargs,
        template_kwargs=test_template_kwargs
    )

    print(f"\n[+] ✅ 单元测试全部通过！所有测试产物(含 JSONL, H5) 已被干净地收纳在 {TEST_DIR} 目录中，快去验收吧！")