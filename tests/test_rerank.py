# tests/test_rerank.py
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.nodes.retrieve import retrieve_node
from src.pipeline.nodes.rerank import rerank_node
from src.models.schemas import GraphState

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
        ("What does E-23 say about model validation?", None),
        ("What are PIPEDA privacy obligations?", "pipeda"),
    ]

    for query, doc_filter in cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Retrieve first
        state = retrieve_node(make_state(query, doc_filter))
        print(f"Retrieved: {len(state['retrieved_chunks'])} chunks")

        # Then rerank
        state = rerank_node(state)
        print(f"Reranked: {len(state['reranked_chunks'])} chunks")
        print(f"\nTop 3 after rerank:")
        for i, c in enumerate(state["reranked_chunks"][:3]):
            print(f"  [{i+1}] doc={c.document_id}")
            print(f"       rerank={c.rerank_score:.4f} | rrf={c.score:.4f}")
            print(f"       {c.chunk_text[:100]}...")