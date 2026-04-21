import os
import json
import torch
from typing import List, Dict, Any
from tqdm import tqdm

# 和你当前 stochastic 脚本保持一致的依赖风格
from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder


class AuxiliaryEvaluator(HiddenStateExtractor):
    """
    只负责 auxiliary self-evaluation：
      - verbalize_response
      - self_evaluator_raw

    不做 stochastic sampling
    不提取 hidden states
    不写 H5
    """

    def generate_single_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """贪婪解码，和你现在 StochasticExtractor 里的行为保持一致。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _load_existing_ids(self, output_jsonl_path: str) -> set:
        """支持断点续跑：如果输出文件已存在，就跳过已经处理过的 sample_id。"""
        existing_ids = set()
        if not os.path.exists(output_jsonl_path):
            return existing_ids

        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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
        """
        从主 metadata 文件中读取样本，单独生成 auxiliary eval 结果。
        输出 JSONL 的每一行形如：
        {
          "sample_id": "...",
          "verbalize_response": "...",        # 若开启
          "self_evaluator_raw": "..."         # 若开启
        }
        """
        if not run_verbalize and not run_self_evaluator:
            raise ValueError("❌ 至少需要开启 run_verbalize 或 run_self_evaluator 中的一个。")

        os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

        print("[*] 正在加载数据池并初始化 Prompt 渲染引擎...")
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
                pbar = tqdm(dataset_items, desc="Auxiliary eval", unit="prompt")
                for item in pbar:
                    sample_id = str(item["sample_id"])

                    if sample_id in existing_ids:
                        skipped_count += 1
                        continue

                    pbar.set_postfix_str(f"sid={sample_id}", refresh=False)

                    # 和你当前 stochastic 流程保持一致：用同一个 PromptBuilder 还原主 prompt
                    prompt_str = prompt_builder.build_prompt(
                        target_item=item,
                        few_shot_pool=dataset_items
                    )

                    out_item = {
                        "sample_id": sample_id
                    }

                    # verbalize：保持你当前 prompt 形式不变
                    if run_verbalize:
                        v_prompt = (
                            f"Question: {prompt_str}\n"
                            f"Answer: {item['model_output_text']}\n"
                            f"Confidence Score (0-1):"
                        )
                        out_item["verbalize_response"] = self.generate_single_response(
                            v_prompt,
                            max_new_tokens=verbalize_max_new_tokens
                        )

                    # self-evaluator：保持你当前 prompt 形式不变
                    if run_self_evaluator:
                        se_prompt = (
                            f"Check consistency.\n"
                            f"Question: {prompt_str}\n"
                            f"Answer: {item['model_output_text']}\n"
                            f"Final Grade (Correct/Incorrect):"
                        )
                        out_item["self_evaluator_raw"] = self.generate_single_response(
                            se_prompt,
                            max_new_tokens=self_eval_max_new_tokens
                        )

                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                    processed_count += 1

        except KeyboardInterrupt:
            print("\n[!] 🛑 接收到打断信号，准备安全退出...")
        finally:
            print(
                f"[+] Auxiliary 生成安全退出！"
                f"新增处理 {processed_count} 条，跳过 {skipped_count} 条已存在样本。"
            )


if __name__ == "__main__":
    evaluator = AuxiliaryEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"trust_remote_code": True, "attn_implementation": "sdpa"}
    )
