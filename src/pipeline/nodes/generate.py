# src/pipeline/nodes/generate.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from langfuse import Langfuse
from src.models.schemas import GraphState, RegulatoryAnswer

load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "generate.txt"
GENERATE_PROMPT = PROMPT_PATH.read_text()

client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    ),
    mode=instructor.Mode.JSON
)

# Optional Langfuse — disabled gracefully when env vars not present
# (eval runner and CI run without Langfuse)
_langfuse = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    _langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )

# Disclaimer trigger words — if any appear in the query, add financial advice disclaimer.
# Rule: cast a wide net. Better to over-trigger than miss a guardrail case.
# Original query (not rewritten) is always checked — rewrite may remove trigger words.
DISCLAIMER_TRIGGERS = {
    # Direct advice requests
    "should i",
    "should we",
    "should our",
    "how should",
    "what should",
    # Recommendation language
    "do you recommend",
    "i recommend",
    "would you recommend",
    "recommend that",
    # Best/optimal framing
    "best way",
    "best approach",
    "best option",
    "best practice",
    "what is the best",
    "what's the best",
    # Financial action language
    "invest",
    "is it worth",
    "should we invest",
    "buy",
    "sell",
    # Advice/guidance requests
    "advise me",
    "advise us",
    "give me advice",
    "your advice",
    # Good idea framing — catches "Is that a good idea?"
    "good idea",
    "is that a good",
    "is this a good",
    # Suggestion language
    "suggest",
    "would you suggest",
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
    query = state["original_query"]
    chunks = state["reranked_chunks"]

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

    # Span wraps the LLM call to capture latency + token counts
    # Only created when Langfuse is configured AND trace_id is available
    trace_id = state.get("langfuse_trace_id")
    span = _langfuse.span(
        trace_id=trace_id,
        name="generate",
        input={"query": query, "context_chunks": len(chunks)},
    ) if (trace_id and _langfuse) else None

    # Instructor wraps the response — get raw completion for token counts
    answer, raw_response = client.chat.completions.create_with_completion(
        model=os.getenv("LITELLM_MODEL", "grok-4.20-0309-non-reasoning"),
        messages=[{"role": "user", "content": prompt}],
        response_model=RegulatoryAnswer,
        temperature=0,
        max_tokens=1000,
    )

    # Reset any disclaimer the LLM may have set directly through Instructor
    # Disclaimer is controlled deterministically by Python, not by the LLM
    answer.disclaimer = None
    answer.requires_human_review = False
    answer = maybe_add_disclaimer(query, answer)

    print(f"  [generate] confidence={answer.confidence}")
    print(f"  [generate] cited_sources={answer.cited_sources}")
    print(f"  [generate] disclaimer={'yes' if answer.disclaimer else 'no'}")

    # Log token counts and output, close span
    if span:
        usage = raw_response.usage
        span.end(
            output={
                "confidence": answer.confidence,
                "cited_sources": answer.cited_sources,
                "disclaimer_triggered": answer.disclaimer is not None,
            },
            metadata={
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "model": os.getenv("LITELLM_MODEL", "grok-4.3"),
            },
        )
        if usage and _langfuse:
            _langfuse.score(trace_id=trace_id, name="prompt_tokens", value=usage.prompt_tokens)
            _langfuse.score(trace_id=trace_id, name="completion_tokens", value=usage.completion_tokens)

    return {**state, "generated_answer": answer}