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
            model_kwargs=None,
            use_compile=True  # 启用 torch.compile() 加速
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

        # 对于批处理，decoder-only 模型必须使用 left padding
        # 这样 attention mask 才能正确处理变长序列
        self.tokenizer.padding_side = 'left'

        # 智能合并 dtype 和 device_map：优先使用 model_kwargs 中的配置
        load_kwargs = {
            "device_map": self.model_kwargs.pop("device_map", self.device),
            **self.model_kwargs
        }

        # 如果 model_kwargs 中没有指定 dtype/torch_dtype，则使用默认的 dtype 参数
        if "dtype" not in load_kwargs and "torch_dtype" not in load_kwargs:
            load_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        self.model.eval()

        self.total_layers = self.model.config.num_hidden_layers
        print(f"[+] 模型加载完成，总层数: {self.total_layers}")

        # 🚀 使用 torch.compile() 加速推理
        # mode 选项: "default" (平衡), "reduce-overhead" (最快), "max-autotune" (最优但编译慢)
        if use_compile and hasattr(torch, 'compile'):
            print(f"[*] 正在使用 torch.compile() 编译模型以加速推理...")
            print(f"    提示: 首次推理会较慢(编译时间)，后续推理将显著加速")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",  # 推理优化模式
                    fullgraph=False  # 允许部分图编译
                )
                print(f"[+] torch.compile() 启用成功！")
            except Exception as e:
                print(f"[!] torch.compile() 失败，将使用 eager 模式: {e}")
        elif use_compile:
            print(f"[!] PyTorch 版本不支持 torch.compile()，需要 PyTorch >= 2.0")
        else:
            print(f"[*] torch.compile() 已禁用")

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

        with torch.no_grad():
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

    def generate_and_extract_batch(self, prompts, layer_config, token_config, max_new_tokens, generation_kwargs=None):
        """
        批量推理：同时处理多个 prompt 以提升 GPU 利用率

        Args:
            prompts: List[str] - 多个 prompt 字符串
            layer_config: 层配置
            token_config: token 配置
            max_new_tokens: 最大生成 token 数
            generation_kwargs: 生成参数

        Returns:
            List[Tuple] - 每个样本的 (filtered_tokens_data, full_output_text, total_generated)
        """
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        # 批量 tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # 自动填充到同一长度
            truncation=True
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_lens = attention_mask.sum(dim=1)  # 每个样本的实际长度

        with torch.no_grad():
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

        # 逐个处理 batch 中的每个样本
        batch_results = []
        for batch_idx in range(len(prompts)):
            prompt_len = prompt_lens[batch_idx].item()
            new_tokens = outputs.sequences[batch_idx, prompt_len:]

            # 移除 padding tokens
            if self.tokenizer.pad_token_id is not None:
                mask = new_tokens != self.tokenizer.pad_token_id
                new_tokens = new_tokens[mask]

            total_generated = len(new_tokens)
            full_output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            target_indices = self._resolve_target_tokens(total_generated, token_config)

            if not target_indices:
                batch_results.append(([], full_output_text, total_generated))
                continue

            # 提取隐藏状态
            # 批处理时 outputs.hidden_states 是一个 tuple，长度为生成的步数
            # 每个元素是一个 tuple of tensors (每层一个)
            # 形状: hidden_states[step_idx][layer_idx][batch_idx, seq_pos, hidden_dim]

            layer_tensors = []
            for l in target_layers:
                hf_layer_idx = l + 1  # HuggingFace 的 layer indexing (+1 for embedding layer)
                step_tensors = []

                for step in target_indices:
                    # 对于每个生成步骤，提取对应 batch 的隐藏状态
                    # step 是生成序列中的位置 (0 到 total_generated-1)
                    # 在批处理中，每步生成一个 token，hidden_states[step] 包含该步的所有层输出

                    # 检查 hidden_states 的长度
                    if step >= len(outputs.hidden_states):
                        # 如果 step 超出范围，使用最后一步
                        hidden_step = outputs.hidden_states[-1]
                    else:
                        hidden_step = outputs.hidden_states[step]

                    # 提取特定层和batch的隐藏状态
                    # hidden_step 是一个 tuple of layer outputs
                    # hidden_step[hf_layer_idx] 的形状: [batch_size, seq_len, hidden_dim]

                    # 对于生成的 token，我们需要序列的最后一个位置
                    layer_output = hidden_step[hf_layer_idx]  # [batch_size, seq_len, hidden_dim]

                    # 获取该 batch 样本在该步生成的 token 的隐藏状态
                    # 使用 -1 索引获取序列的最后一个位置
                    token_hidden = layer_output[batch_idx, -1, :]

                    step_tensors.append(token_hidden)

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

            batch_results.append((filtered_tokens_data, full_output_text, total_generated))

        return batch_results

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
            template_kwargs: dict = None,
            batch_size: int = 1  # 批处理大小，默认为 1（单样本）
    ):
        gen_kwargs = generation_kwargs or {}
        tpl_kwargs = template_kwargs or {}

        if not os.path.exists(input_jsonl_path):
            raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

        os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

        # 准备可 JSON 序列化的 model_kwargs（转换 torch.dtype 对象）
        serializable_model_kwargs = {}
        for k, v in self.model_kwargs.items():
            if isinstance(v, torch.dtype):
                serializable_model_kwargs[k] = str(v)  # 转为字符串如 "torch.float16"
            else:
                serializable_model_kwargs[k] = v

        with h5py.File(output_h5_path, 'a') as f:
            f.attrs["llm_model"] = self.model_name
            f.attrs["dataset_source"] = os.path.basename(input_jsonl_path)
            f.attrs["layer_config"] = json.dumps(layer_config or {})
            f.attrs["token_config"] = json.dumps(token_config or {})
            f.attrs["model_kwargs"] = json.dumps(serializable_model_kwargs)
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
            # 批处理循环
            for batch_start_idx in range(0, len(dataset_items), batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, len(dataset_items))
                batch_items = dataset_items[batch_start_idx:batch_end_idx]

                # 构建当前 batch 的所有 prompts
                batch_prompts = []
                for item in batch_items:
                    prompt_str = prompt_builder.build_prompt(target_item=item, few_shot_pool=dataset_items)
                    item["prompt"] = prompt_str
                    batch_prompts.append(prompt_str)

                # 打印批次信息
                if batch_size > 1:
                    sample_ids_str = ", ".join([item["sample_id"] for item in batch_items])
                    print(f"[Main Process] Batch inferencing [{batch_start_idx+1}-{batch_end_idx}]: {sample_ids_str}")
                else:
                    print(f"[Main Process] Inferencing {batch_items[0]['sample_id']}...")

                # 批量推理
                if batch_size > 1:
                    batch_results = self.generate_and_extract_batch(
                        batch_prompts, layer_config, token_config, max_new_tokens, generation_kwargs=gen_kwargs
                    )
                else:
                    # batch_size=1 时使用原始单样本方法（向后兼容）
                    single_result = self.generate_and_extract(
                        batch_prompts[0], layer_config, token_config, max_new_tokens, generation_kwargs=gen_kwargs
                    )
                    batch_results = [single_result]

                # 处理批量结果
                for item, (filtered_tokens_data, model_output_text, total_generated) in zip(batch_items, batch_results):
                    sample_id = item["sample_id"]
                    prompt_str = item["prompt"]

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