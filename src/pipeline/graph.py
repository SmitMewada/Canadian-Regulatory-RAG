# src/pipeline/graph.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.models.schemas import GraphState
from src.pipeline.nodes.query_rewrite import query_rewrite_node
from src.pipeline.nodes.retrieve import retrieve_node
from src.pipeline.nodes.rerank import rerank_node
from src.pipeline.nodes.generate import generate_node
from src.pipeline.nodes.inline_eval import inline_eval_node
from src.pipeline.nodes.citation_check import citation_check_node

load_dotenv()

def should_retry(state: GraphState) -> str:
    """
    Conditional edge — decides whether to retry or proceed to output.
    Retry only once (retry_count == 0) and only if eval score is low.
    """
    score = state.get("inline_eval_score")
    retry_count = state.get("retry_count", 0)

    if score is not None and score < 3.0 and retry_count == 0:
        print(f"  [graph] low eval score ({score}) — retrying with expanded retrieval")
        return "retry"
    return "proceed"

def retry_node(state: GraphState) -> GraphState:
    """
    Expands retrieval top_k and re-runs retrieve → rerank → generate.
    Called only once — prevents infinite loops.
    """
    print(f"  [retry] expanding retrieval and retrying...")
    return {
        **state,
        "retry_count": state["retry_count"] + 1,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "generated_answer": None,
        "inline_eval_score": None,
    }

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("inline_eval", inline_eval_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("retry", retry_node)

    # Entry point
    graph.set_entry_point("query_rewrite")

    # Linear edges
    graph.add_edge("query_rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "inline_eval")

    # Conditional edge after eval — retry or proceed
    graph.add_conditional_edges(
        "inline_eval",
        should_retry,
        {
            "retry": "retry",
            "proceed": "citation_check",
        }
    )

    # Retry loops back to retrieve with expanded top_k
    graph.add_edge("retry", "retrieve")

    # Citation check is the final node
    graph.add_edge("citation_check", END)

    return graph.compile()


# Compiled graph — import this elsewhere
pipeline = build_graph()


def run_pipeline(query: str, document_filter: str = None) -> dict:
    """
    Main entry point. Takes a query, returns the final state.
    """
    initial_state = GraphState(
        original_query=query,
        rewritten_query="",
        document_filter=document_filter,
        retrieved_chunks=[],
        reranked_chunks=[],
        generated_answer=None,
        inline_eval_score=None,
        citation_valid=None,
        retry_count=0,
        langfuse_trace_id=None,
        cache_hit=False,
    )

    final_state = pipeline.invoke(initial_state)
    return final_state