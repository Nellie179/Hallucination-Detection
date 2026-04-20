import os
import json
import argparse
from typing import Dict, Any, Optional
from datetime import datetime, date
from datasets import load_dataset, get_dataset_split_names


# ==========================================
# 0. 辅助函数 - JSON 序列化工具
# ==========================================
def sanitize_for_json(obj: Any) -> Any:
    """
    递归地将对象中的非JSON可序列化类型（如 datetime, PIL Image）转换为字符串。
    这样可以安全地保存 HuggingFace datasets 中的原始数据。
    """
    # 处理 datetime 对象
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # 处理 PIL Image 对象（通过检查类名，避免导入 PIL）
    elif hasattr(obj, '__class__') and 'Image' in obj.__class__.__name__:
        # 返回图片的描述信息而不是图片本身
        try:
            mode = getattr(obj, 'mode', 'unknown')
            size = getattr(obj, 'size', (0, 0))
            return f"<PIL.Image mode={mode} size={size[0]}x{size[1]}>"
        except:
            return "<PIL.Image object>"

    # 递归处理字典和列表
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    # 其他类型保持不变
    else:
        return obj


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



############## New Added
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
        # TheoremQA 使用首字母大写的字段名: Question, Answer, Answer_type
        q = row.get("Question") or row.get("question") or row.get("problem") or ""
        theorem = row.get("theorem") or row.get("theorem_name") or row.get("topic") or ""

        # 将定理作为前置 Context 传入
        ctx = f"Relevant Theorem/Topic: {theorem}" if theorem else ""

        # 获取答案，优先使用首字母大写的 Answer
        answer = str(row.get("Answer") or row.get("answer") or row.get("final_answer") or row.get("solution") or "")

        return {
            "task_type": "reasoning",  # 触发 prompt_builder 里的 step-by-step
            "system_instruction": "Apply the relevant mathematical or scientific theorem to solve the problem.",
            "context": ctx,
            "question": q,
            "choices": {},
            "ground_truths": [answer] if answer else [""],
            "incorrect_answers": []
        }


class MATHAdapter(BaseDatasetAdapter):
    """
    MATH: 竞赛级数学题。
    终极方案：使用社区合并的完整版源，一次性拿下 7500 条训练集。
    """
    # 🎯 指向全量合并的备份源
    dataset_path = "JeremiahZ/hendrycks_math_merged"
    dataset_name = "default"  # 或者设为 None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # 字段和官方完全一致
        problem = row.get("problem") or ""
        solution = row.get("solution") or ""
        
        return {
            "task_type": "reasoning", 
            "system_instruction": "Solve the mathematical problem step by step. Put your final answer in \\boxed{}.",
            "context": f"Level: {row.get('level', 'unknown')}, Type: {row.get('type', 'unknown')}",
            "question": problem,
            "choices": {},
            "ground_truths": [solution],
            "incorrect_answers": []
        }

class SVAMPAdapter(BaseDatasetAdapter):
    """
    SVAMP: A Challenge Set for Elementary-Level Math Word Problems.
    特点：通过改变题目结构来测试模型的鲁棒性。包含 700 条 Train 和 300 条 Test。
    """
    # 🎯 社区中最常用且格式规整的 SVAMP 源
    dataset_path = "ChilleD/SVAMP"
    dataset_name = "default"  # 如果报错可以改为 None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # SVAMP 把题目拆成了 Body(背景) 和 Question(问题)
        body = row.get("Body") or ""
        question = row.get("Question") or ""
        
        # 拼接成完整的题目
        full_problem = f"{body} {question}".strip()
        
        # 提取答案并确保转换为字符串
        solution = str(row.get("Answer", "")).strip()
        
        return {
            "task_type": "reasoning", 
            "system_instruction": "Solve the mathematical problem step by step. Put your final answer in \\boxed{}.",
            "context": f"Type: {row.get('Type', 'unknown')}", # 记录题目类型（如 Addition, Subtraction 等）
            "question": full_problem,
            "choices": {},
            "ground_truths": [solution],
            "incorrect_answers": []
        }

class ASDivAdapter(BaseDatasetAdapter):
    """
    ASDiv: Academia Sinica Diverse MWP Dataset.
    特点：题型和语言模式极其多样化，难度适中，非常适合做小样本或零样本推理测试。
    HuggingFace源: EleutherAI/asdiv
    """
    # 🎯 社区中最权威、维护最好的 ASDiv 源
    dataset_path = "EleutherAI/asdiv"
    dataset_name = "asdiv"  # 如果 HF 报错可改为 None 或 "main"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # ASDiv 同样把题目拆成了 Body(背景) 和 Question(问题)
        body = row.get("body", "").strip()
        question = row.get("question", "").strip()
        
        # 拼接成完整的题目
        full_problem = f"{body} {question}".strip()
        
        # ⚠️ 关键清洗：ASDiv 的答案通常自带单位，格式如 "14 (apples)" 或 "12"
        # 咱们必须把纯数字提取出来，劈开括号取前半部分
        raw_answer = str(row.get("answer", "")).strip()
        solution = raw_answer.split(" (")[0].strip() if " (" in raw_answer else raw_answer
        
        return {
            "task_type": "reasoning", 
            "system_instruction": "Solve the mathematical problem step by step. Put your final answer in \\boxed{}.",
            "context": f"Formula: {row.get('formula', 'unknown')}", # 记录它底层所依赖的公式
            "question": full_problem,
            "choices": {},
            "ground_truths": [solution],
            "incorrect_answers": []
        }

# --- 注册表 ---
DATASET_REGISTRY = {
    "truthful_qa": TruthfulQAAdapter,
    "halueval_qa": HaluEvalQAAdapter,
    "trivia_qa": TriviaQAAdapter,
    "coqa": CoQAAdapter,
    "squad_v2": SQuADv2Adapter,
    "arc_challenge": ARCChallengeAdapter,
    "xsum": XSumAdapter,
    "gsm8k": GSM8KAdapter,
    "human_eval": HumanEvalAdapter,
    "xlam_agent": XLamFunctionCallingAdapter,
    "mbpp": MBPPAdapter,
    ############################################
    "hotpotqa": HotpotQAAdapter,
    "commonsenseqa": CommonsenseQAAdapter,
    "mmlu": MMLUAdapter,
    "belebele": BelebeleAdapter,
    "ragtruth": RAGTruthAdapter, #256
    "theoremqa": TheoremQAAdapter, #1024
    "math": MATHAdapter,
    "svamp": SVAMPAdapter
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
                "original_doc": sanitize_for_json(row)  # 无损保留原始数据，方便后期追溯（将datetime等对象转为字符串）
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