import os
import json
import argparse
from typing import Dict, Any, Optional
import random
from datasets import load_dataset, get_dataset_split_names


# ==========================================
# 1. 适配器基类 (Adapter Base) - 纯粹的 Schema 转换器
# ==========================================
class BaseDatasetAdapter:
    """数据集适配器基类，负责将极其混乱的原始数据，映射为绝对统一的 Universal Schema"""
    dataset_path: str = ""
    dataset_name: str = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心输出规范：
        必须返回包含以下键的字典 (没有的值填空字符串 "" 或空列表 [])：
        - task_type (str): 任务类型 (qa, multiple_choice, summarization, reasoning)
        - system_instruction (str): 特殊指令
        - context (str): 背景材料/文章
        - question (str): 核心提问
        - choices (dict): 选项字典 (如 {"A": "xx", "B": "yy"})
        - ground_truths (list): 所有可接受的正确答案
        - incorrect_answers (list): 陷阱/已知错误答案
        """
        raise NotImplementedError


# ==========================================
# 2. 具体数据集实现 (彻底解耦，拥抱结构化)
# ==========================================
class TruthfulQAAdapter(BaseDatasetAdapter):
    dataset_path = "truthful_qa"
    dataset_name = "generation"

    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": row['correct_answers'],  # 完美：原生就是一个 List
            "incorrect_answers": row['incorrect_answers']  # 完美：原生陷阱 List
        }


class HaluEvalQAAdapter(BaseDatasetAdapter):
    dataset_path = "pminervini/HaluEval"
    dataset_name = "qa"
    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "Please answer the question based strictly on the provided context.",
            "context": row['knowledge'],
            "question": row['question'],
            "choices": {},
            "ground_truths": [row['right_answer']],
            "incorrect_answers": [row.get('hallucinated_answer', "")]  # 捕获极其宝贵的幻觉对照样本
        }


class TriviaQAAdapter(BaseDatasetAdapter):
    dataset_path = "trivia_qa"
    dataset_name = "rc.nocontext"

    def extract_structured_data(self, row):
        # TriviaQA 非常贴心，自带了一个 aliases 列表，包含了该答案的所有同义词写法！
        valid_answers = row['answer'].get('aliases', [row['answer']['value']])
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": valid_answers,
            "incorrect_answers": []
        }


class CoQAAdapter(BaseDatasetAdapter):
    dataset_path = "coqa"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": row['story'],
            "question": row['questions'][0],  # 提取第一轮提问
            "choices": {},
            "ground_truths": [row['answers']['input_text'][0]],
            "incorrect_answers": []
        }


class CoQAMultiTurnAdapter(BaseDatasetAdapter):
    """
    CoQA 多轮对话适配器 (CoQA-MultiTurn)
    动态抽取某一轮作为当前问题，并将该轮之前的所有问答作为 Conversation History 拼接进 Context 中。
    测试目标：模型能否结合原始 Story 和历史对话上下文，准确回答具有指代关系的新问题。
    """
    dataset_path = "coqa"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        story = row['story']
        questions = row['questions']
        answers = row['answers']['input_text']
        
        num_turns = len(questions)
        
        # 边界保护：如果该样本只有 1 轮对话，退化为基础 QA
        if num_turns <= 1:
            return {
                "task_type": "qa",
                "system_instruction": "Read the story carefully and answer the question.",
                "context": f"Story:\n{story}",
                "question": questions[0],
                "choices": {},
                "ground_truths": [answers[0]],
                "incorrect_answers": []
            }
        
        # 核心逻辑：随机挑选一个大于 0 的轮次作为“当前要考核的问题”
        # 这样能保证必然存在至少 1 轮的历史对话
        target_turn_idx = random.randint(1, num_turns - 1)
        
        # 拼装历史对话轨迹
        history_blocks = []
        for i in range(target_turn_idx):
            history_blocks.append(f"User: {questions[i]}\nAssistant: {answers[i]}")
            
        history_str = "\n\n".join(history_blocks)
        
        # 将故事与历史轨迹组装成一个巨大的复合 Context
        composite_context = (
            f"=== Background Story ===\n{story}\n\n"
            f"=== Conversation History ===\n{history_str}"
        )
        
        return {
            "task_type": "qa",
            "system_instruction": "Based on the Background Story and the Conversation History, answer the user's latest question accurately.",
            "context": composite_context,
            "question": questions[target_turn_idx],  # 考核当前轮次问题
            "choices": {},
            "ground_truths": [answers[target_turn_idx]], # 考核当前轮次答案
            "incorrect_answers": []
        }

class SQuADv2Adapter(BaseDatasetAdapter):
    dataset_path = "squad_v2"
    dataset_name = None

    def extract_structured_data(self, row):
        # SQuAD 的 text 字段本身就是一个包含了多个人工标注答案的 List
        answers = row['answers']['text']
        if len(answers) == 0:
            answers = ["I don't know.", "Unanswerable"]  # 陷阱题的标准化处理

        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": row['context'],
            "question": row['question'],
            "choices": {},
            "ground_truths": answers,
            "incorrect_answers": []
        }


class ARCChallengeAdapter(BaseDatasetAdapter):
    dataset_path = "ai2_arc"
    dataset_name = "ARC-Challenge"

    def extract_structured_data(self, row):
        # 极其优雅的选项组装，生成形如 {"A": "xx", "B": "yy"} 的字典
        choices_dict = {
            label: text for label, text in zip(row['choices']['label'], row['choices']['text'])
        }
        return {
            "task_type": "multiple_choice",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": choices_dict,
            "ground_truths": [row['answerKey']],  # 正确答案是单个字母，如 "A"
            "incorrect_answers": []
        }


class XSumAdapter(BaseDatasetAdapter):
    dataset_path = "xsum"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "summarization",
            "system_instruction": "",
            "context": row['document'],
            "question": "Please summarize the above document in one sentence.",
            "choices": {},
            "ground_truths": [row['summary']],
            "incorrect_answers": []
        }


class GSM8KAdapter(BaseDatasetAdapter):
    dataset_path = "gsm8k"
    dataset_name = "main"

    def extract_structured_data(self, row):
        return {
            "task_type": "reasoning",
            # 直接在 Schema 级别注入指令，供 PromptBuilder 识别
            "system_instruction": "Please think step by step and provide the final answer at the end.",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": [row['answer']],
            "incorrect_answers": []
        }

class HumanEvalAdapter(BaseDatasetAdapter):
    """
    OpenAI HumanEval (代码补全)
    探测模型在生成逻辑代码、理解算法边界条件时的隐藏层状态。
    """
    dataset_path = "openai_humaneval"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "coding",
            # 动态注入针对代码题的系统指令
            "system_instruction": "You are an expert Python developer. Complete the provided Python function based on the docstring. Only output valid Python code.",
            "context": "",
            "question": row['prompt'], # 这里面包含了 import 和 def xxx(): 以及注释
            "choices": {},
            "ground_truths": [row['canonical_solution']], # 标准答案代码
            "incorrect_answers": []
        }


class XLamFunctionCallingAdapter(BaseDatasetAdapter):
    """
    Salesforce xLAM Function Calling 数据集。
    测试模型在看到 API 列表后，能否准确理解用户意图并输出正确的 API 调用 JSON。
    """
    dataset_path = "Salesforce/xlam-function-calling-60k"
    dataset_name = None

    def extract_structured_data(self, row):
        # row['tools'] 是一个包含了工具描述的 JSON 字符串
        # row['query'] 是用户的自然语言提问
        # row['answers'] 是期望模型输出的工具调用指令 (也是 JSON 字符串)

        return {
            "task_type": "agent_action",
            "system_instruction": "You are a helpful assistant with access to various tools. Based on the User's question, select the appropriate tool from the Context and output the exact tool call in JSON format. If no tool is needed, answer directly.",
            "context": f"Available Tools:\n{row['tools']}",
            "question": row['query'],
            "choices": {},
            "ground_truths": [row['answers']],  # GPT-4 裁判会完美判断生成的 JSON 是否等价
            "incorrect_answers": []
        }


class MBPPAdapter(BaseDatasetAdapter):
    """
    Google MBPP (基础 Python 编程题)
    测试模型基础的算法生成和逻辑实现能力。
    """
    dataset_path = "mbpp"
    dataset_name = "full"

    def extract_structured_data(self, row):
        # row['text'] 是英文的编程要求
        # row['code'] 是标准的 Python 代码答案
        # row['test_list'] 是用于验证代码的 assert 语句列表

        # 我们可以把测试用例放进 Context 里作为额外的约束和提示
        tests_str = "\n".join(row['test_list'])

        return {
            "task_type": "coding",
            "system_instruction": "You are an expert Python programmer. Write a Python function to solve the problem. Your code must pass the provided assertion tests.",
            "context": f"Your code must pass the following tests:\n{tests_str}",
            "question": row['text'],
            "choices": {},
            "ground_truths": [row['code']],
            "incorrect_answers": []
        }


import string
from typing import Dict, Any

# ==========================================
# 新增数据集适配器 (适配 Universal Schema)
# ==========================================

class HaluBenchAdapter(BaseDatasetAdapter):
    """
    HaluBench: 包含 Context-Question-Answer 三元组以及幻觉标签。
    HF: PatronusAI/HaluBench
    精妙之处：如果是幻觉，直接打入 incorrect_answers 作为已知陷阱，供裁判模型捕获。
    """
    dataset_path = "PatronusAI/HaluBench"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ctx = row.get("context") or row.get("Context") or row.get("passage") or ""
        q = row.get("question") or row.get("Question") or ""
        ans = row.get("answer") or row.get("Answer") or row.get("reference_answer") or row.get("ground_truth") or ""
        
        # 提取幻觉标签
        label_raw = row.get("is_hallucination", row.get("hallucination", row.get("label")))
        is_hallucinated = str(label_raw).lower() in ["true", "1", "yes"]

        ground_truths = []
        incorrect_answers = []
        
        if is_hallucinated:
            incorrect_answers.append(ans)
        else:
            ground_truths.append(ans)

        return {
            "task_type": "qa",
            "system_instruction": "Please answer the question based strictly on the provided context.",
            "context": ctx,
            "question": q,
            "choices": {},
            "ground_truths": ground_truths,
            "incorrect_answers": incorrect_answers
        }


class HotpotQAAdapter(BaseDatasetAdapter):
    """
    HotpotQA: 多跳 QA，需要跨段落聚合证据。
    HF: hotpotqa/hotpot_qa
    """
    dataset_path = "hotpotqa/hotpot_qa"
    dataset_name = "fullwiki"  # 或者 "distractor"

    def _flatten_context(self, ctx: Any, max_sent_per_title: int = 8) -> str:
        """展开嵌套的 Context 字典或列表"""
        if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
            blocks = []
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                blocks.append(f"[{title}] " + " ".join(sents[:max_sent_per_title]))
            return "\n".join(blocks)

        if isinstance(ctx, list):
            blocks = []
            for item in ctx:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    title, sents = item, item
                    if isinstance(sents, list):
                        blocks.append(f"[{title}] " + " ".join(sents[:max_sent_per_title]))
            if blocks:
                return "\n".join(blocks)
        return "" if ctx is None else str(ctx)

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        context_text = self._flatten_context(row.get("context"))
        
        return {
            "task_type": "qa",
            "system_instruction": "Answer the question by synthesizing information from the multiple provided contexts.",
            "context": context_text,
            "question": row.get("question", ""),
            "choices": {},
            "ground_truths": [row.get("answer", "")],
            "incorrect_answers": []
        }


class CommonsenseQAAdapter(BaseDatasetAdapter):
    """
    CommonsenseQA: 常识推理多选题。
    HF: tau/commonsense_qa
    """
    dataset_path = "tau/commonsense_qa"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        choices_raw = row.get("choices", {})
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        
        choices_dict = {str(lab): str(txt) for lab, txt in zip(labels, texts)}
        
        return {
            "task_type": "multiple_choice",
            "system_instruction": "Use your common sense to select the most appropriate option.",
            "context": "",
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [row.get("answerKey", "")],
            "incorrect_answers": []
        }


class MMLUAdapter(BaseDatasetAdapter):
    """
    MMLU: 多学科大规模多选题考试。
    HF: cais/mmlu
    """
    dataset_path = "cais/mmlu"
    dataset_name = "all"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        choices_list = row.get("choices") or row.get("options") or []
        
        # 动态生成 A, B, C, D...
        choices_dict = {}
        for i, opt in enumerate(choices_list):
            label = chr(65 + i) if i < 26 else str(i) # 65 is 'A'
            choices_dict[label] = str(opt)

        ans_raw = row.get("answer")
        ground_truth_label = ""
        
        # 原生 MMLU 的 answer 通常是 integer index (0-3)
        if isinstance(ans_raw, int) and ans_raw < len(choices_list):
            ground_truth_label = chr(65 + ans_raw)
        else:
            ground_truth_label = str(ans_raw)

        subject = row.get("subject", "").replace("_", " ").title()
        instruction = f"This is a multiple-choice question about {subject}." if subject else ""

        return {
            "task_type": "multiple_choice",
            "system_instruction": instruction,
            "context": "",
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [ground_truth_label],
            "incorrect_answers": []
        }


class BelebeleAdapter(BaseDatasetAdapter):
    """
    Belebele: 多语言阅读理解。
    HF: facebook/belebele
    """
    dataset_path = "facebook/belebele"
    dataset_name = "eng_Latn"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        passage = row.get("passage") or row.get("context") or row.get("flores_passage") or ""
        
        # Belebele 通常把选项存在 mc_answer1 到 mc_answer4，或者直接以列表存在 choices
        options = []
        for k in ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4", "answer0", "answer1", "answer2", "answer3"]:
            if k in row and row[k]:
                options.append(row[k])
                
        if not options and "choices" in row:
             options = row["choices"]

        choices_dict = {chr(65 + i): str(opt) for i, opt in enumerate(options)}

        # Belebele 的 correct_answer_num 通常是 "1", "2", "3", "4" (1-based index)
        ans_raw = row.get("correct_answer_num", "")
        ground_truth_label = ""
        try:
            # 尝试将 1-based 的字符串索引转为 A, B, C, D
            idx = int(ans_raw) - 1
            if 0 <= idx < len(options):
                ground_truth_label = chr(65 + idx)
        except (ValueError, TypeError):
            ground_truth_label = str(row.get("label") or row.get("answer") or "")

        return {
            "task_type": "multiple_choice",
            "system_instruction": "Read the passage carefully and select the correct answer.",
            "context": passage,
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [ground_truth_label],
            "incorrect_answers": []
        }


class RAGTruthAdapter(BaseDatasetAdapter):
    """
    RAGTruth (processed): 针对 RAG 场景的幻觉检测数据集。
    HF: wandb/RAGTruth-processed
    """
    dataset_path = "wandb/RAGTruth-processed"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ctx = row.get("context") or row.get("retrieved_context") or row.get("retrieved_docs") or ""
        prompt = row.get("prompt") or row.get("query") or row.get("question") or ""
        response = row.get("response") or row.get("generated_response") or row.get("output") or ""
        
        label = row.get("label") or row.get("is_hallucination")
        is_hallucinated = str(label).lower() in ["true", "1", "yes"]

        ground_truths = []
        incorrect_answers = []
        if is_hallucinated:
            incorrect_answers.append(response)
        else:
            ground_truths.append(response)

        return {
            "task_type": "qa",
            "system_instruction": "Evaluate or synthesize based on the retrieved context.",
            "context": ctx,
            "question": prompt,
            "choices": {},
            "ground_truths": ground_truths,
            "incorrect_answers": incorrect_answers
        }


class TheoremQAAdapter(BaseDatasetAdapter):
    """
    TheoremQA: 定理驱动的 QA（数学/物理等复杂推理）。
    HF: TIGER-Lab/TheoremQA
    """
    dataset_path = "TIGER-Lab/TheoremQA"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        q = row.get("question") or row.get("problem") or ""
        theorem = row.get("theorem") or row.get("theorem_name") or row.get("topic") or ""
        
        # 将定理作为前置 Context 传入
        ctx = f"Relevant Theorem/Topic: {theorem}" if theorem else ""
        
        return {
            "task_type": "reasoning",  # 触发 prompt_builder 里的 step-by-step
            "system_instruction": "Apply the relevant mathematical or scientific theorem to solve the problem.",
            "context": ctx,
            "question": q,
            "choices": {},
            "ground_truths": [str(row.get("answer") or row.get("final_answer") or row.get("solution") or "")],
            "incorrect_answers": []
        }


class MATHAdapter(BaseDatasetAdapter):
    """
    MATH: 竞赛级数学题。
    HF: HuggingFaceH4/MATH (或 lighteval/MATH)
    """
    dataset_path = "HuggingFaceH4/MATH"
    dataset_name = "default"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("solution") or row.get("answer") or ""
        
        return {
            "task_type": "reasoning",  # 触发 CoT
            "system_instruction": "Solve the mathematical problem step by step. Put your final answer in \\boxed{}.",
            "context": "",
            "question": problem,
            "choices": {},
            "ground_truths": [solution], # 将整个推导过程丢给 GPT-4 裁判去判别核心答案
            "incorrect_answers": []
        }

# --- 注册表 ---
DATASET_REGISTRY = {
    "truthful_qa": TruthfulQAAdapter,
    "halueval_qa": HaluEvalQAAdapter,
    "trivia_qa": TriviaQAAdapter,
    "coqa": CoQAAdapter,      # 单轮对话
    "squad_v2": SQuADv2Adapter,
    "arc_challenge": ARCChallengeAdapter,
    "xsum": XSumAdapter,
    "gsm8k": GSM8KAdapter,
    "human_eval": HumanEvalAdapter,
    "xlam_agent": XLamFunctionCallingAdapter,
    "mbpp": MBPPAdapter,
    ##############################
    "coqa_multiturn": CoQAMultiTurnAdapter, # 新版多轮
    "halubench": HaluBenchAdapter,
    "hotpotqa": HotpotQAAdapter,
    "commonsenseqa": CommonsenseQAAdapter,
    "mmlu": MMLUAdapter,
    "belebele": BelebeleAdapter,
    "ragtruth": RAGTruthAdapter,
    "theoremqa": TheoremQAAdapter,
    "math": MATHAdapter,
}


# ==========================================
# 3. 核心处理与导出逻辑
# ==========================================
def process_dataset(
        adapter_name: str,
        output_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None
):
    if adapter_name not in DATASET_REGISTRY:
        raise ValueError(f"未找到数据集适配器: {adapter_name}. 可用的有: {list(DATASET_REGISTRY.keys())}")

    adapter = DATASET_REGISTRY[adapter_name]()

    # --- 自适应 Split 嗅探与回退机制 ---
    print(f"[*] 正在探测数据集 {adapter.dataset_path} 的可用 Split...")
    try:
        available_splits = get_dataset_split_names(adapter.dataset_path, adapter.dataset_name)

        if split not in available_splits:
            print(f"[!] 警告: 目标 split '{split}' 不存在。该数据集仅有: {available_splits}")
            fallback_priority = ["train", "validation", "test", "data"]
            new_split = None

            for fb in fallback_priority:
                if fb in available_splits:
                    new_split = fb
                    break

            if not new_split and len(available_splits) > 0:
                new_split = available_splits[0]

            print(f"[*] 🚀 已触发自适应机制，自动回退使用 split: '{new_split}'")
            split = new_split

    except Exception as e:
        print(f"[!] 探测 Split 失败，将强制尝试加载 '{split}'。错误信息: {e}")

    print(f"[*] 正在通过 HuggingFace 加载 {adapter.dataset_path} ({split}集)...")
    dataset = load_dataset(adapter.dataset_path, adapter.dataset_name, split=split)
    dataset = dataset.shuffle(seed=42)

    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{adapter_name}_{split}.jsonl"
    out_filepath = os.path.join(output_dir, out_filename)

    print(f"[*] 正在格式化数据并导出至 {out_filepath} ...")

    processed_count = 0
    with open(out_filepath, 'w', encoding='utf-8') as f:
        for i, row in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break

            # 提取高度结构化的纯净数据
            structured_data = adapter.extract_structured_data(row)

            # 组装最终 JSON 行
            item = {
                "sample_id": f"{adapter_name}_{split}_{i:06d}",
                "structured_data": structured_data,
                "original_doc": row  # 无损保留原始数据，方便后期追溯
            }

            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"[+] 成功处理并保存 {processed_count} 条数据！")

    if max_samples is not None and processed_count < max_samples:
        print(f"[!] 提示: 你请求了 {max_samples} 条数据，但该 Split 已被榨干。实际仅输出 {processed_count} 条。")
    print("\n")
    return out_filepath


# ==========================================
# 4. CLI 接口与单元测试
# ==========================================
def main():
    DEFAULT_DATASET = "gsm8k"
    DEFAULT_OUTPUT_DIR = "./unified_datasets"
    DEFAULT_SPLIT = "validation"
    DEFAULT_MAX_SAMPLES = 5

    parser = argparse.ArgumentParser(description="基于 Universal Schema 的大一统数据处理器")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"注册表中的数据集名称 (默认: {DEFAULT_DATASET})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出 JSONL 的目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        help=f"数据集的 split, 如 train/validation/test (默认: {DEFAULT_SPLIT})")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"最大处理条数，填 0 则处理全量 (默认: {DEFAULT_MAX_SAMPLES})")

    args = parser.parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None

    process_dataset(
        adapter_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=max_samples
    )


if __name__ == "__main__":
    main()