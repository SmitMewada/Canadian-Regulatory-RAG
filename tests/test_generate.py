# tests/test_generate.py
from dotenv import load_dotenv

from src.pipeline.nodes.retrieve import retrieve_node
from src.pipeline.nodes.rerank import rerank_node
from src.pipeline.nodes.generate import generate_node
from src.models.schemas import GraphState

load_dotenv()

def make_state(query: str, doc_filter: str = None) -> GraphState:
    return GraphState(
        original_query=query,
        rewritten_query=query,
        document_filter=doc_filter,
        retrieved_chunks=[],
        reranked_chunks=[],
        generated_answer=None,
        inline_eval_score=None,
        citation_valid=None,
        retry_count=0,
        langfuse_trace_id=None,
        cache_hit=False,
    )

if __name__ == "__main__":
    cases = [
        ("What does OSFI E-23 require for model validation?", None),
        ("What are the privacy obligations under PIPEDA?", "pipeda"),
        ("Should our bank use AI for credit decisions?", None),   # guardrail test
        ("What are SEC requirements for algorithmic trading?", None),  # out of scope
    ]

    for query, doc_filter in cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        state = make_state(query, doc_filter)
        state = retrieve_node(state)
        state = rerank_node(state)
        state = generate_node(state)

        answer = state["generated_answer"]
        print(f"Answer: {answer.answer[:300]}")
        print(f"Sources: {answer.cited_sources}")
        print(f"Confidence: {answer.confidence}")
        print(f"Disclaimer: {answer.disclaimer is not None}")
        print(f"Needs review: {answer.requires_human_review}")