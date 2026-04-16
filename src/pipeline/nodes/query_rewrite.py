# src/pipeline/nodes/query_rewrite.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.models.schemas import GraphState

load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rewrite.txt"
REWRITE_PROMPT = PROMPT_PATH.read_text()

# xAI is OpenAI-compatible — just swap the base_url
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.x.ai/v1",
)

def query_rewrite_node(state: GraphState) -> GraphState:
    """
    Rewrites the user query for better retrieval.
    Preserves regulatory identifiers (E-23, PIPEDA, B-13, etc.)
    """
    query = state["original_query"]

    # If query contains a specific section reference,
    # skip rewriting — don't risk losing precision
    PRESERVE_PATTERNS = ["section", "clause", "§", "4.", "3.", "2."]
    if any(p in query.lower() for p in PRESERVE_PATTERNS):
        return {**state, "rewritten_query": query}

    response = client.chat.completions.create(
        model=os.getenv("LITELLM_MODEL", "grok-4-fast"),  # no xai/ prefix needed here
        messages=[
            {"role": "user", "content": REWRITE_PROMPT.format(query=query)}
        ],
        temperature=0,
        max_tokens=200,
    )

    rewritten = response.choices[0].message.content.strip()
    return {**state, "rewritten_query": rewritten}