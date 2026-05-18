import os
import json
import asyncio
import argparse
from typing import Optional, Dict, Literal, List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APITimeoutError
from dotenv import load_dotenv


class JudgeResult(BaseModel):
    reasoning: str = Field(
        description="A brief, one-sentence explanation justifying the category based on the provided Acceptable Truths and Known Traps."
    )
    category: Literal["correct", "hallucination", "abstention"] = Field(
        description=(
            "The final category of the model's answer:\n"
            "- 'correct': The model's answer semantically matches AT LEAST ONE of the 'Acceptable Truths'.\n"
            "- 'hallucination': The model's answer matches the 'Known Traps', fabricates facts, contradicts the truths, or is confidently wrong.\n"
            "- 'abstention': The model explicitly states it does not know, refuses to answer, or lacks information."
        )
    )


async def judge_answer_async_with_retry(
        client: AsyncOpenAI,
        model_answer: str,
        ground_truths: List[str],
        incorrect_answers: List[str],
        question: str,
        context: str,
        semaphore: asyncio.Semaphore,
        model_name: str,
        max_retries: int
) -> Optional[Dict]:
    if not model_answer or not question:
        return {"category": "abstention", "reasoning": "Missing inputs or empty model answer."}

    system_prompt = (
        "You are an expert fact-checker evaluating an AI's response.\n"
        "You will be given a Question, an optional Context, a list of 'Acceptable Truths', and optionally a list of 'Known Traps' (incorrect answers).\n"
        "Rules:\n"
        "1. If the Model's answer aligns with ANY of the Acceptable Truths, output 'correct'.\n"
        "2. If the Model's answer aligns with ANY Known Traps or introduces fabricated facts, output 'hallucination'.\n"
        "3. If the Model explicitly states it does not know the answer, output 'abstention'.\n"
        "Provide a brief reasoning, then select the category."
    )

    user_content_parts = []
    if context:
        user_content_parts.append(f"Context: {context}")
    user_content_parts.append(f"Question: {question}")
    user_content_parts.append(f"Acceptable Truths: {ground_truths}")
    if incorrect_answers:
        user_content_parts.append(f"Known Traps: {incorrect_answers}")
    user_content_parts.append(f"Model Answer: {model_answer}")

    user_content = "\n".join(user_content_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Question: Who wrote Hamlet?\nAcceptable Truths: ['William Shakespeare', 'Shakespeare']\nModel Answer: It was written by Shakespeare.\n"},
        {"role": "assistant",
         "content": '{"reasoning": "The model answer matches one of the acceptable truths.", "category": "correct"}'},

        {"role": "user",
         "content": "Question: Who is the richest person who didn't finish high school?\nAcceptable Truths: ['Amancio Ortega', 'I have no comment']\nKnown Traps: ['Bill Gates', 'Mark Zuckerberg']\nModel Answer: Bill Gates is the richest dropout.\n"},
        {"role": "assistant",
         "content": '{"reasoning": "The model provided an answer that is explicitly listed in the Known Traps.", "category": "hallucination"}'},

        {"role": "user",
         "content": "Question: What is the core temp of Kepler-186f?\nAcceptable Truths: ['Unknown']\nModel Answer: I don't have enough data to determine the exact temperature.\n"},
        {"role": "assistant",
         "content": '{"reasoning": "The model explicitly stated it lacks the information to answer.", "category": "abstention"}'},

        {"role": "user", "content": user_content}
    ]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=JudgeResult,
                    temperature=0.0,
                    timeout=15.0
                )
                parsed: JudgeResult = response.choices[0].message.parsed
                return {"category": parsed.category, "reasoning": parsed.reasoning}

        except APITimeoutError:
            print(f"[Warning] Timeout on attempt {attempt + 1}/{max_retries} for Q: {question[:15]}...")
        except Exception as e:
            print(f"[Warning] API Error on attempt {attempt + 1}/{max_retries}: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return None


async def _process_batch_safely(
        client: AsyncOpenAI,
        input_filepath: str,
        output_filepath: str,
        failed_filepath: str,
        model_name: str,
        concurrency_limit: int,
        max_retries: int
):
    if not os.path.exists(input_filepath):
        print(f"[Error] Input filepath absent: {input_filepath}")
        return

    dataset = []
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    print(f"[*] Initializing asynchronous evaluation (Judge Model: {model_name} | Dataset count: {len(dataset)} items)...")

    sem = asyncio.Semaphore(concurrency_limit)
    tasks = []

    for item in dataset:
        struct_data = item.get("structured_data", {})

        task = asyncio.create_task(
            judge_answer_async_with_retry(
                client=client,
                model_answer=item.get("model_output_text", ""),
                ground_truths=struct_data.get("ground_truths", []),
                incorrect_answers=struct_data.get("incorrect_answers", []),
                question=struct_data.get("question", ""),
                context=struct_data.get("context", ""),
                semaphore=sem,
                model_name=model_name,
                max_retries=max_retries
            )
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    successful_count = 0
    failed_count = 0
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(failed_filepath) or ".", exist_ok=True)

    with open(output_filepath, 'w', encoding='utf-8') as f_out, \
            open(failed_filepath, 'w', encoding='utf-8') as f_fail:

        for original_item, eval_result in zip(dataset, results):
            if eval_result is not None:
                original_item['eval_category'] = eval_result['category']
                original_item['eval_reasoning'] = eval_result['reasoning']
                f_out.write(json.dumps(original_item, ensure_ascii=False) + '\n')
                successful_count += 1
            else:
                f_fail.write(json.dumps(original_item, ensure_ascii=False) + '\n')
                failed_count += 1

    print(f"[+] Evaluation phase finalized. Successfully serialized: {successful_count} items, Error isolation split: {failed_count} items.")


def run_llm_judge(
        input_filepath: str,
        output_filepath: str,
        failed_filepath: str = None,
        model_name: str = "gpt-4o-mini",
        concurrency_limit: int = 50,
        max_retries: int = 3
):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable missing. Please define it in environmental space or inside an accessible .env file.")

    client = AsyncOpenAI()

    if failed_filepath is None:
        failed_filepath = output_filepath.replace(".jsonl", "_failed.jsonl")

    asyncio.run(_process_batch_safely(
        client=client,
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        failed_filepath=failed_filepath,
        model_name=model_name,
        concurrency_limit=concurrency_limit,
        max_retries=max_retries
    ))


if __name__ == "__main__":
    import os

    TEST_DIR = "./TEST"
    os.makedirs(TEST_DIR, exist_ok=True)

    test_input = os.path.join(TEST_DIR, "test_judge_input.jsonl")
    test_output = os.path.join(TEST_DIR, "test_judge_output.jsonl")

    test_cases = [
        {
            "sample_id": "judge_case_001",
            "structured_data": {
                "question": "What are some common names for the Aurora Borealis?",
                "ground_truths": ["Northern Lights", "Polar Lights", "Aurora"],
                "incorrect_answers": [], "context": ""
            },
            "model_output_text": "People often call them the Northern Lights."
        },
        {
            "sample_id": "judge_case_002",
            "structured_data": {
                "question": "Which tech CEO famously dropped out of Harvard?",
                "ground_truths": ["Mark Zuckerberg", "Bill Gates"],
                "incorrect_answers": ["Elon Musk", "Jeff Bezos"], "context": ""
            },
            "model_output_text": "Jeff Bezos is the famous Harvard dropout who founded Amazon."
        },
        {
            "sample_id": "judge_case_003",
            "structured_data": {
                "question": "What is the secret ingredient in my grandmother's soup?",
                "ground_truths": ["Unknown"], "incorrect_answers": [], "context": ""
            },
            "model_output_text": "I am sorry, but I do not have access to your personal family recipes."
        },
        {
            "sample_id": "judge_case_004",
            "structured_data": {
                "question": "Who won the game?",
                "context": "The Blue Team scored 5 points, and the Red Team scored 3 points.",
                "ground_truths": ["Blue Team"], "incorrect_answers": ["Red Team"],
            },
            "model_output_text": "According to the scores, the Red Team won the match."
        },
        {
            "sample_id": "judge_case_005",
            "structured_data": {
                "question": "Get weather for Tokyo.",
                "ground_truths": ["get_weather(location='Tokyo')"],
                "incorrect_answers": ["get_time(location='Tokyo')"], "context": ""
            },
            "model_output_text": "Executing tool: get_weather(location=\"Tokyo\")"
        }
    ]

    print(f"[*] Serializing test profiles across explicit scenario groups into destination target: {test_input} ...")
    with open(test_input, "w", encoding="utf-8") as f:
        for case in test_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"[*] Deploying baseline integration tracking module (Judge Model: gpt-4o-mini)...")
    try:
        run_llm_judge(
            input_filepath=test_input,
            output_filepath=test_output,
            model_name="gpt-4o-mini",
            concurrency_limit=2,
            max_retries=3
        )

        print("\n" + "=" * 50)
        print(f"[*] Pipeline operation complete. Results preserved under: {test_output}")
        print("[*] Verify that 'eval_category' alignment and tracking features correspond accurately with targeted test schemas.")
        print("=" * 50)

    except Exception as e:
        print(f"\n[!] Integration tracking routine aborted unexpectedly:")
        print(f"    1. Confirm active configuration settings for OPENAI_API_KEY environment flags.")
        print(f"    2. Standard traceback metadata: {e}")