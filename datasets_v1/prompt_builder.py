import random
from typing import Dict, List, Any
from transformers import AutoTokenizer

class LLMPromptBuilder:
    def __init__(
            self,
            model_name: str,
            global_system_prompt: str = "You are a helpful, accurate, and honest AI assistant.",
            num_shots: int = 0,
            template_kwargs: dict = None
    ):
        self.model_name = model_name
        self.global_system_prompt = global_system_prompt
        self.num_shots = num_shots
        self.template_kwargs = template_kwargs or {}

        print(f"[*] [PromptBuilder] Loading tokenizer instance for official chat template resolution: {model_name}")
        if self.template_kwargs:
            print(f"    - Forwarding chat template arguments: {self.template_kwargs}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.chat_template is None:
            print("[!] Warning: Model configuration is missing an official chat template. Engaging standard fallback block.")
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'].upper() + ': ' + message['content'] + '\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
            )

    def _render_user_content(self, structured_data: Dict[str, Any]) -> str:
        task_type = structured_data.get("task_type", "qa")
        instruction = structured_data.get("system_instruction", "").strip()
        context = structured_data.get("context", "").strip()
        question = structured_data.get("question", "").strip()
        choices = structured_data.get("choices", None)

        content_parts = []

        if instruction:
            content_parts.append(f"Instruction: {instruction}\n")

        if context:
            prefix = "Document:" if task_type == "summarization" else "Context:"
            content_parts.append(f"{prefix}\n{context}\n")

        content_parts.append(f"Question: {question}")

        if choices and isinstance(choices, dict) and len(choices) > 0:
            content_parts.append("Options:")
            for label, text in choices.items():
                content_parts.append(f"{label}. {text}")
            content_parts.append("\nPlease strictly answer with the correct option letter(s).")

        elif task_type == "reasoning":
            content_parts.append("\nPlease think step by step and provide the final answer at the end.")

        return "\n".join(content_parts)

    def build_prompt(self, target_item: Dict[str, Any], few_shot_pool: List[Dict[str, Any]] = None) -> str:
        messages = [{"role": "system", "content": self.global_system_prompt}]

        if self.num_shots > 0 and few_shot_pool:
            candidates = [ex for ex in few_shot_pool if ex["sample_id"] != target_item["sample_id"]]
            shots = random.sample(candidates, min(self.num_shots, len(candidates)))

            for shot in shots:
                shot_data = shot["structured_data"]
                user_text = self._render_user_content(shot_data)

                if shot_data["ground_truths"]:
                    assistant_text = str(shot_data["ground_truths"][0])
                else:
                    assistant_text = "I don't know."

                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})

        target_user_text = self._render_user_content(target_item["structured_data"])
        messages.append({"role": "user", "content": target_user_text})

        final_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            **self.template_kwargs
        }

        final_prompt_string = self.tokenizer.apply_chat_template(
            messages,
            **final_template_kwargs
        )

        return final_prompt_string


if __name__ == "__main__":
    import os
    builder = LLMPromptBuilder(model_name="meta-llama/Llama-3.2-1B-Instruct")

    mock_pool = [
        {"sample_id": "qa_001", "structured_data": {"task_type": "qa", "system_instruction": "", "context": "", "question": "What is the capital of Japan?", "choices": {}, "ground_truths": ["Tokyo"], "incorrect_answers": ["Kyoto"]}},
        {"sample_id": "qa_002", "structured_data": {"task_type": "qa", "system_instruction": "Answer strictly based on the context.", "context": "The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel.", "question": "Who built it?", "choices": {}, "ground_truths": ["Gustave Eiffel"], "incorrect_answers": []}},
        {"sample_id": "mc_001", "structured_data": {"task_type": "multiple_choice", "system_instruction": "", "context": "", "question": "Which planet is known as the Red Planet?", "choices": {"A": "Earth", "B": "Mars", "C": "Jupiter"}, "ground_truths": ["B"], "incorrect_answers": []}},
        {"sample_id": "math_001", "structured_data": {"task_type": "reasoning", "system_instruction": "", "context": "", "question": "If I have 5 apples and eat 2, how many are left?", "choices": {}, "ground_truths": ["3"], "incorrect_answers": []}},
        {"sample_id": "code_001", "structured_data": {"task_type": "coding", "system_instruction": "You are an expert Python coder.", "context": "", "question": "def add(a, b):\n    \"\"\"Return sum\"\"\"\n", "choices": {}, "ground_truths": ["    return a + b"], "incorrect_answers": []}},
        {"sample_id": "agent_001", "structured_data": {"task_type": "agent_action", "system_instruction": "Select the right API.", "context": "APIs:\n1. get_weather(loc)", "question": "Weather in Paris?", "choices": {}, "ground_truths": ["get_weather('Paris')"], "incorrect_answers": []}}
    ]

    print("\n" + "="*70)
    print("🚀 Prompt Builder: 0-Shot Task Routing Evaluation")
    print("="*70)
    builder.num_shots = 0
    for target in [mock_pool[2], mock_pool[5]]:
        print(f"\n>>> Target Task Type Context: {target['structured_data']['task_type']} <<<")
        print(builder.build_prompt(target_item=target, few_shot_pool=mock_pool))

    print("\n" + "="*70)
    print("🚀 Prompt Builder: 2-Shot Dynamic Cross-Task Few-shot Evaluation")
    print("="*70)
    builder.num_shots = 2
    target_coding = mock_pool[4]
    print(f"\n>>> Target Task Type Context: {target_coding['structured_data']['task_type']} (With 2 random few-shot samples) <<<")
    print(builder.build_prompt(target_item=target_coding, few_shot_pool=mock_pool))
    print("-" * 70)