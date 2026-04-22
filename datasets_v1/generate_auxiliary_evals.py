import os
import json
import torch
from typing import List, Dict, Any

# 完全对齐你的主干依赖
from hidden_state import HiddenStateExtractor
from prompt_builder import LLMPromptBuilder

class AuxiliaryEvaluator(HiddenStateExtractor):
    """
    只负责 auxiliary self-evaluation：
      - verbalize_response
      - self_evaluator_raw

    不做 stochastic sampling，不提取 hidden states
    🚀 [重构版]：全面支持全局依赖注入，与 HiddenStateExtractor 底层推理逻辑严格对齐！
    """

    def __init__(self, model_name=None, model=None, tokenizer=None, model_kwargs=None, **kwargs):
        """支持从大管家直接注入模型，或者兜底自我加载"""
        if model is not None and tokenizer is not None:
            self.model_name = model_name or "injected_model"
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
            self.model_kwargs = model_kwargs or {}
            
            # 补齐父类属性
            if hasattr(self.model.config, "num_hidden_layers"):
                self.total_layers = self.model.config.num_hidden_layers
            else:
                self.total_layers = self.model.config.text_config.num_hidden_layers

            print(f"[*] AuxiliaryEvaluator 成功接收指挥官注入的模型实例 (设备: {self.device})")
        else:
            # 严格调用父类 HiddenStateExtractor 的初始化
            super().__init__(model_name=model_name, model_kwargs=model_kwargs, **kwargs)

    def _generate_text(self, chat_messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        """
        🚀 核心生成器：严密对齐 HiddenStateExtractor.generate_and_extract 的 Tensor 构建方式
        彻底解决 shape 报错，并压制无关 warning。
        """
        # 1. 渲染标准 Chat 文本 (绝不硬拼接)
        prompt_str = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Tokenize 构建 inputs
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        # 3. Generate 推理 (与你 main pipeline 的传参方式一模一样)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # 显式关闭，压制警告
                top_p=None,        # 显式关闭，压制警告
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 4. Decode 截取新生成的 Token
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
                for item in dataset_items:
                    sample_id = str(item["sample_id"])

                    if sample_id in existing_ids:
                        skipped_count += 1
                        continue

                    print(f"[Main] Auxiliary 判决生成中: {sample_id}...")

                    out_item = {"sample_id": sample_id}

                    # ==========================================
                    # 🚀 终极解法：调用 _render_user_content 提取【纯净文本】
                    # 绝对防止 <|begin_of_text|> 等特殊符产生嵌套污染！
                    # ==========================================
                    clean_question = prompt_builder._render_user_content(item["structured_data"])
                    clean_answer = item.get('model_output_text', '')

                    # 1. Verbalize (自信度评估)
                    if run_verbalize:
                        v_messages = [
                            {"role": "system", "content": "You are a strict evaluator. Respond ONLY with a single float number between 0.0 and 1.0 representing your confidence in the provided answer. Do not output any other text."},
                            {"role": "user", "content": f"Question: {clean_question}\n\nProposed Answer: {clean_answer}\n\nHow confident are you that this answer is completely correct? Score from 0.0 to 1.0:"}
                        ]
                        out_item["verbalize_response"] = self._generate_text(v_messages, verbalize_max_new_tokens)

                    # 2. Self Evaluator (正误自评)
                    if run_self_evaluator:
                        se_messages = [
                            {"role": "system", "content": "You are a strict teacher grading a test. You must reply with either 'Correct' or 'Incorrect' at the very end of your response."},
                            {"role": "user", "content": f"Question: {clean_question}\n\nProposed Answer: {clean_answer}\n\nEvaluate the proposed answer. Is it True or False? Final Grade (Correct/Incorrect):"}
                        ]
                        out_item["self_evaluator_raw"] = self._generate_text(se_messages, self_eval_max_new_tokens)

                    f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                    processed_count += 1
                    
                    # 保护显存：万条数据无忧
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n[!] 🛑 接收到打断信号，准备安全退出...")
        finally:
            print(f"[+] Auxiliary 生成安全退出！新增处理 {processed_count} 条，跳过 {skipped_count} 条已存在样本。")

if __name__ == "__main__":
    evaluator = AuxiliaryEvaluator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"trust_remote_code": True, "attn_implementation": "sdpa"}
    )