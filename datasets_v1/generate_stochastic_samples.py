import os
import json
import time
import h5py
import torch
import math
import ml_dtypes
import multiprocessing as mp
from typing import Dict, Any, List

from transformers import LogitsProcessor, LogitsProcessorList

from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder


class MultinomialSafetyProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=-100.0, posinf=100.0, neginf=-100.0)
        return scores


def stochastic_hdf5_writer_worker(data_queue, h5_filepath, jsonl_filepath, extract_hs):
    print(f"[Writer Process] Stochastic sample disk synchronization process initiated (PID: {os.getpid()}).")
    h5_ctx = h5py.File(h5_filepath, 'a') if extract_hs else None

    with open(jsonl_filepath, 'a', encoding='utf-8') as f_json:
        while True:
            item = data_queue.get()
            if item is None:
                break

            sample_id = item["sample_id"]
            try:
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

                f_json.write(json.dumps(item["meta_item"], ensure_ascii=False) + '\n')
                f_json.flush()
                if h5_ctx:
                    h5_ctx.flush()

            except Exception as e:
                print(f"[Writer Process] IO Serialization failure for sample {sample_id}: {e}")

    if h5_ctx:
        h5_ctx.close()
    print(f"[Writer Process] IO serialization process safely detached.")


class StochasticExtractor(HiddenStateExtractor):

    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None):
        if model is not None and tokenizer is not None:
            self.model_name = model_name or "injected_model"
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.model_kwargs = model_kwargs or {}

            if hasattr(self.model.config, "num_hidden_layers"):
                self.total_layers = self.model.config.num_hidden_layers
            else:
                self.total_layers = self.model.config.text_config.num_hidden_layers

            print(
                f"[*] StochasticExtractor successfully initialized with injected model instance on device: {self.device}")
        else:
            super().__init__(model_name, model_kwargs)

    def generate_and_extract_stochastic(self, prompt, layer_config, token_config, max_new_tokens, num_samples,
                                        generation_kwargs=None):
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        batch_results = []
        safety_processors = LogitsProcessorList([MultinomialSafetyProcessor()])

        for i in range(num_samples):
            with torch.no_grad():
                final_gen_kwargs = {
                    **gen_kwargs,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "num_return_sequences": 1,
                    "return_dict_in_generate": True,
                    "output_hidden_states": True,
                    "output_scores": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "renormalize_logits": True,
                    "logits_processor": safety_processors
                }
                outputs = self.model.generate(**final_gen_kwargs)

                transition_scores = None
                if hasattr(outputs, "scores"):
                    if hasattr(outputs, "beam_indices") and outputs.beam_indices is not None:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences, outputs.scores, beam_indices=outputs.beam_indices, normalize_logits=True
                        )
                    else:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )

            new_tokens = outputs.sequences[0, prompt_len:]

            stop_masks = (new_tokens == self.tokenizer.eos_token_id) | (new_tokens == self.tokenizer.pad_token_id)
            stop_indices = stop_masks.nonzero(as_tuple=True)[0]
            total_generated = stop_indices[0].item() if len(stop_indices) > 0 else len(new_tokens)

            valid_tokens = new_tokens[:total_generated]
            full_output_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)

            token_logprobs = []
            if transition_scores is not None and total_generated > 0:
                for s in transition_scores[0, :total_generated].cpu().numpy():
                    val = float(s)
                    if math.isinf(val) or math.isnan(val):
                        val = -15.0
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

        existing_ids = set()
        if os.path.exists(output_jsonl_path):
            with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            existing_ids.add(str(json.loads(line)["sample_id"]))
                        except:
                            pass

        if existing_ids:
            print(
                f"[*] Recovery manager tracking: Found {len(existing_ids)} completed historical items. Skipping records...")

        data_queue = mp.Queue(maxsize=max_queue_size)
        writer_process = mp.Process(
            target=stochastic_hdf5_writer_worker,
            args=(data_queue, output_h5_path, output_jsonl_path, extract_stochastic_hs)
        )
        writer_process.daemon = True
        writer_process.start()

        print("[*] Loading historical sequence datasets and initializing structural template engine...")
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
        skipped_count = 0
        try:
            for item in dataset_items:
                sample_id = str(item["sample_id"])

                if sample_id in existing_ids:
                    skipped_count += 1
                    continue

                print(f"[Main] Running multi-turn stochastic generational sampling for sequence: {sample_id}...")

                try:
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
                except Exception as e:
                    print(
                        f"\n[!] Unexpected runtime exception for sequence item {sample_id}. Diverting into Dead Letter Queue (DLQ). Log exception: {e}")
                    error_log_path = output_jsonl_path + ".failed_ids.txt"
                    with open(error_log_path, 'a', encoding='utf-8') as f_err:
                        f_err.write(f"{sample_id}\t{e}\n")
                    torch.cuda.empty_cache()
                    continue

        except KeyboardInterrupt:
            print(
                "\n[!] Execution cycle interrupted via system interrupt command. Shutting down worker pathways safely...")
        finally:
            data_queue.put(None)
            writer_process.join(timeout=10)
            if writer_process.is_alive():
                writer_process.terminate()
            print(
                f"[+] Generative tracking cycle terminated safely. Bypassed {skipped_count} existing tracks, serialized {processed_count} execution sequences.")