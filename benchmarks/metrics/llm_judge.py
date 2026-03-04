"""
LLM Judge for LoCoMo Benchmark.

Uses Gemini 2.5 Flash to evaluate if generated answers are correct.
This matches the LLM-as-a-Judge (J) metric used in memory benchmarks.
"""
import os
import re
import json
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("xmem.benchmark.llm_judge")

LLM_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


_judge_model: Optional[ChatGoogleGenerativeAI] = None


def get_judge_model() -> ChatGoogleGenerativeAI:
    """Get or create the judge model."""
    global _judge_model
    if _judge_model is None:
        _judge_model = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
            thinking_level="medium",
        )
    return _judge_model


def _parse_judge_response(content: str) -> float:
    """Parse judge response and return 1.0 for CORRECT, 0.0 for WRONG."""
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
            else:
                parts.append(str(c))
        content = "\n".join(parts)
    
    json_match = re.search(r'\{[^}]+\}', content)
    if json_match:
        try:
            result = json.loads(json_match.group())
            label = result.get("label", "UNKNOWN")
            return 1.0 if label == "CORRECT" else 0.0
        except json.JSONDecodeError:
            pass
    
    content_upper = content.upper()
    if "CORRECT" in content_upper and "WRONG" not in content_upper:
        return 1.0
    elif "WRONG" in content_upper:
        return 0.0
    
    return 0.0


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    model: Optional[ChatGoogleGenerativeAI] = None
) -> float:
    """
    Use LLM to judge if the answer is correct.
    
    Args:
        question: The question asked
        gold_answer: Expected ground truth answer
        generated_answer: Model's generated answer
        model: Optional custom model (uses default if None)
        
    Returns:
        1.0 if CORRECT, 0.0 if WRONG
    """
    if model is None:
        model = get_judge_model()
    
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=str(generated_answer)
    )
    
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        return _parse_judge_response(response.content)
    except Exception as e:
        logger.error(f"LLM Judge error: {e}")
        return 0.0


async def evaluate_llm_judge_async(
    question: str,
    gold_answer: str,
    generated_answer: str,
    model: Optional[ChatGoogleGenerativeAI] = None
) -> float:
    """
    Async version of LLM judge evaluation.
    
    Args:
        question: The question asked
        gold_answer: Expected ground truth answer
        generated_answer: Model's generated answer
        model: Optional custom model (uses default if None)
        
    Returns:
        1.0 if CORRECT, 0.0 if WRONG
    """
    if model is None:
        model = get_judge_model()
    
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=str(generated_answer)
    )
    
    try:
        response = await model.ainvoke([{"role": "user", "content": prompt}])
        return _parse_judge_response(response.content)
    except Exception as e:
        logger.error(f"LLM Judge async error: {e}")
        return 0.0
