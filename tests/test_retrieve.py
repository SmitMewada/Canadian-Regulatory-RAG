# tests/test_retrieve.py
from dotenv import load_dotenv
from src.pipeline.nodes.retrieve import retrieve_node
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
        ("What does E-23 say about model validation?", None),
        ("What are PIPEDA privacy obligations?", "pipeda"),
        ("What are SEC requirements for AI?", None),   # out of scope
    ]

    for query, doc_filter in cases:
        print(f"\nQuery: {query}")
        print(f"Filter: {doc_filter or 'none'}")
        result = retrieve_node(make_state(query, doc_filter))
        chunks = result["retrieved_chunks"]
        print(f"Retrieved: {len(chunks)} chunks")
        for i, c in enumerate(chunks[:3]):
            print(f"  [{i+1}] doc={c.document_id} score={c.score:.4f} | {c.chunk_text[:80]}...")


