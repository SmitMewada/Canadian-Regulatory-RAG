import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.models.schemas import GraphState
from src.pipeline.nodes.generate import build_context

load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "inline_eval.txt"
INLINE_EVAL_PROMPT = PROMPT_PATH.read_text()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.x.ai/v1",
)

INLINE_EVAL_THRESHOLD = float(os.getenv("INLINE_EVAL_THRESHOLD", "3.0"))

def inline_eval_node(state: GraphState) -> GraphState:
    """
    LLM-as-judge: scores whether the generated answer is
    supported by the retrieved context. Score 1-5.
    If score < threshold (default 3): flag low confidence.
    Max 1 retry — don't loop indefinitely.
    """
    answer = state["generated_answer"]
    chunks = state["reranked_chunks"] or state["retrieved_chunks"]

    # Nothing to evaluate
    if not answer or not chunks:
        print("  [inline_eval] skipped — no answer or chunks")
        return {**state, "inline_eval_score": None}

    # Skip eval for "I don't know" answers — they're always faithful
    if "don't have sufficient information" in answer.answer:
        print("  [inline_eval] skipped — IDK answer, always faithful")
        return {**state, "inline_eval_score": 5.0}

    context = build_context(chunks)
    prompt = INLINE_EVAL_PROMPT.format(
        context=context,
        answer=answer.answer,
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL", "grok-4-fast"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,  # only needs a single digit
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw)
        score = max(1.0, min(5.0, score))  # clamp to 1-5 range
    except (ValueError, Exception) as e:
        print(f"  [inline_eval] parse error: {e} — defaulting to 3.0")
        score = 3.0

    print(f"  [inline_eval] score={score}/5 threshold={INLINE_EVAL_THRESHOLD}")

    # If score below threshold and no retry used yet — flag it
    if score < INLINE_EVAL_THRESHOLD and state["retry_count"] == 0:
        print(f"  [inline_eval] score below threshold — flagging low confidence")
        answer.confidence = "low"
        answer.requires_human_review = True

    return {**state, "inline_eval_score": score, "generated_answer": answer}