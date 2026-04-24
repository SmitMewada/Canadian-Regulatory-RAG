# src/pipeline/nodes/generate.py
import os
from pathlib import Path
from dotenv import load_dotenv
from litellm import query
from openai import OpenAI
import instructor
from src.models.schemas import GraphState, RegulatoryAnswer

load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "generate.txt"
GENERATE_PROMPT = PROMPT_PATH.read_text()

# Patch OpenAI client with Instructor for structured output
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
)

DISCLAIMER_TRIGGERS = {
    "should i", "should we", "should our",
    "do you recommend", "what should",
    "advise me", "advise us",
    "is it worth", "should we invest",
}

def build_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_ref = chunk.document_id.upper().replace("_", " ")
        heading = f" — {chunk.section_heading}" if chunk.section_heading else ""
        page = f" (p.{chunk.page_number})" if chunk.page_number else ""
        parts.append(
            f"[Source {i}: {doc_ref}{heading}{page}]\n{chunk.chunk_text}"
        )
    return "\n\n".join(parts)

def maybe_add_disclaimer(query: str, answer: RegulatoryAnswer) -> RegulatoryAnswer:
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in DISCLAIMER_TRIGGERS):
        answer.disclaimer = (
            "This system provides information about Canadian financial regulations. "
            "It does not constitute financial, legal, or compliance advice. "
            "Consult qualified professionals for decisions."
        )
        answer.requires_human_review = True
    return answer

def generate_node(state: GraphState) -> GraphState:
    """
    Generates a structured answer from reranked chunks.
    Uses Instructor + Pydantic to enforce output schema.
    Temperature=0 for compliance — no creativity, only grounded answers.
    """
    query = state["original_query"]
    chunks = state["reranked_chunks"]

    # Fall back to retrieved if rerank was skipped
    if not chunks:
        chunks = state["retrieved_chunks"]

    if not chunks:
        answer = RegulatoryAnswer(
            answer="I don't have sufficient information in the available regulatory documents to answer this question.",
            cited_sources=[],
            confidence="low",
            requires_human_review=False,
        )
        return {**state, "generated_answer": answer}

    context = build_context(chunks)
    prompt = GENERATE_PROMPT.format(context=context, query=query)

    print(f"  [generate] query='{query[:60]}...'")
    print(f"  [generate] context chunks={len(chunks)}")

    answer = client.chat.completions.create(
    model=os.getenv("LITELLM_MODEL", "grok-4-fast"),
    messages=[{"role": "user", "content": prompt}],
    response_model=RegulatoryAnswer,
    temperature=0,
    max_tokens=1000,
)

# Reset — Python controls disclaimer, not the LLM
    answer.disclaimer = None
    answer.requires_human_review = False
    answer = maybe_add_disclaimer(query, answer)

    print(f"  [generate] confidence={answer.confidence}")
    print(f"  [generate] cited_sources={answer.cited_sources}")
    print(f"  [generate] disclaimer={'yes' if answer.disclaimer else 'no'}")

    return {**state, "generated_answer": answer}