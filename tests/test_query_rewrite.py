# tests/test_query_rewrite.py
import os
from src.pipeline.nodes.query_rewrite import query_rewrite_node
from src.models.schemas import GraphState
from dotenv import load_dotenv

load_dotenv()

def make_state(query: str) -> GraphState:
    return GraphState(
        original_query=query,
        rewritten_query="",
        document_filter=None,
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
        ("What does E-23 say about model validation?", "normal rewrite"),
        ("Should our bank use AI for credit scoring?", "guardrail — should still rewrite"),
        ("What does section 4.2 require?", "has 'section' — should NOT rewrite"),
        ("What are PIPEDA obligations for AI systems?", "normal rewrite"),
    ]

    for query, description in cases:
        result = query_rewrite_node(make_state(query))
        changed = result["rewritten_query"] != query
        print(f"[{description}]")
        print(f"  Original:  {query}")
        print(f"  Rewritten: {result['rewritten_query']}")
        print(f"  Modified:  {changed}\n")