import os
import json
import torch
from typing import List, Dict, Any

from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder


class AuxiliaryEvaluator(HiddenStateExtractor):

    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None, **kwargs):
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
                f"[*] AuxiliaryEvaluator successfully initialized with injected model instance on device: {self.device}")
        else:
            super().__init__(model_name=model_name, model_kwargs=model_kwargs, **kwargs)

    def _generate_text(self, chat_messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        prompt_str = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id
            )

        new_tokens = outputs[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _load_existing_ids(self, output_jsonl_path: str) -> set:
        existing_ids = set()
        if not os.path.exists(output_jsonl_path):
            return existing_ids
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    if "sample_id" in d:
                        existing_ids.add(str(d["sample_id"]))
                except Exception:
                    continue
        return existing_ids

    def process_auxiliary_from_file(
            self,
            input_jsonl_path: str,
            output_jsonl_path: str,
            system_prompt: str,
            num_shots: int,
            template_kwargs: dict,
            run_verbalize: bool = True,
            run_self_evaluator: bool = False,
            verbalize_max_new_tokens: int = 10,
            self_eval_max_new_tokens: int = 150,
            overwrite: bool = False,
    ):
        if not run_verbalize and not run_self_evaluator:
            raise ValueError("At least one evaluation protocol must be activated: run_verbalize or run_self_evaluator.")

        os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

        print("[*] Initializing prompt serialization pipeline and dataset handlers...")
        dataset_items = self._load_jsonl(input_jsonl_path)

        prompt_builder = LLMPromptBuilder(
            model_name=self.model_name,
            global_system_prompt=system_prompt,
            num_shots=num_shots,
            template_kwargs=template_kwargs
        )

        if overwrite and os.path.exists(output_jsonl_path):
            os.remove(output_jsonl_path)

        existing_ids = set() if overwrite else self._load_existing_ids(output_jsonl_path)
        mode = "a" if os.path.exists(output_jsonl_path) else "w"

        processed_count = 0
        skipped_count = 0

        try:
            with open(output_jsonl_path, mode, encoding="utf-8") as f_out:
                for item in dataset_items:
                    sample_id = str(item["sample_id"])

                    if sample_id in existing_ids:
                        skipped_count += 1
                        continue

                    print(f"[Main] Generating auxiliary evaluation trace for sample: {sample_id}...")

                    out_item = {"sample_id": sample_id}

                    clean_question = prompt_builder._render_user_content(item["structured_data"])
                    clean_answer = item.get('model_output_text', '')

                    if run_verbalize:
                        v_messages = [
                            {"role": "system",
                             "content": "You are a strict evaluator. Respond ONLY with a single float number between 0.0 and 1.0 representing your confidence in the provided answer. Do not output any other text."},
                            {"role": "user",
                             "content": f"Question: {clean_question}\n\nProposed Answer: {clean_answer}\n\nHow confident are you that this answer is completely correct? Score from 0.0 to 1.0:"}
                        ]
                        out_item["verbalize_response"] = self._generate_text(v_messages, verbalize_max_new_tokens)

                    if run_self_evaluator:
                        se_messages = [
                            {"role": "system",
                             "content": "You are a strict teacher grading a test. You must reply with either 'Correct' or 'Incorrect' at the very end of your response."},
                            {"role": "user",
                             "content": f"Question: {clean_question}\n\nProposed Answer: {clean_answer}\n\nEvaluate the proposed answer. Is it True or False? Final Grade (Correct/Incorrect):"}
                        ]
                        out_item["self_evaluator_raw"] = self._generate_text(se_messages, self_eval_max_new_tokens)

                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                    processed_count += 1

                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n[!] Execution interrupted via system catch, securing files and exiting safely...")
        finally:
            print(
                f"[+] Auxiliary processing complete. Generated {processed_count} new item scores, skipped {skipped_count} identical records.")


if __name__ == "__main__":
    evaluator = AuxiliaryEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"trust_remote_code": True, "attn_implementation": "sdpa"}
    )