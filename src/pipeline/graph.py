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
from src.pipeline.cache import get_cached_response, store_cached_response
import uuid
from langfuse.callback import CallbackHandler
from langfuse import Langfuse


load_dotenv()

_langfuse_client = None
 
_langfuse_pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
_langfuse_sec = os.environ.get("LANGFUSE_SECRET_KEY")
 
if _langfuse_pub and _langfuse_sec:
    _langfuse_client = Langfuse(
        public_key=_langfuse_pub,
        secret_key=_langfuse_sec,
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )
else:
    print("[graph] Langfuse not configured — tracing disabled. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable.")


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


def run_pipeline_patched(query: str, document_filter: str = None) -> dict:
    """
    Patched version of run_pipeline with optional Langfuse tracing.
    Paste this over your existing run_pipeline() in graph.py.
    """
    session_id = f"session-{uuid.uuid4()}"
 
    # --- CACHE HIT PATH ---
    cached = get_cached_response(query)
    if cached:
        if _langfuse_client:
            trace = _langfuse_client.trace(
                name="regulatory_rag_query",
                session_id=session_id,
                input={"query": query},
                metadata={"cache_hit": True, "document_filter": document_filter},
            )
            trace.score(name="cache_hit", value=1)
            _langfuse_client.flush()
            trace_id = trace.id
        else:
            trace_id = None
 
        return {
            "original_query": query,
            "rewritten_query": query,
            "document_filter": document_filter,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "generated_answer": cached,
            "inline_eval_score": None,
            "citation_valid": None,
            "retry_count": 0,
            "langfuse_trace_id": trace_id,
            "cache_hit": True,
        }
 
    # --- CACHE MISS — FULL PIPELINE ---
    if _langfuse_client:
        trace = _langfuse_client.trace(
            name="regulatory_rag_query",
            session_id=session_id,
            input={"query": query},
            metadata={"cache_hit": False, "document_filter": document_filter},
        )
        handler = CallbackHandler(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
            session_id=session_id,
        )
        config = {"callbacks": [handler]}
        trace_id = trace.id
    else:
        trace = None
        config = {}
        trace_id = None
 
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
        langfuse_trace_id=trace_id,
        cache_hit=False,
    )
 
    final_state = pipeline.invoke(initial_state, config=config)
 
    answer = final_state.get("generated_answer")
 
    if trace and _langfuse_client:
        trace.update(
            output={"answer": answer.answer if answer else None},
            metadata={
                "inline_eval_score": final_state.get("inline_eval_score"),
                "citation_valid": final_state.get("citation_valid"),
                "retry_count": final_state.get("retry_count"),
                "confidence": answer.confidence if answer else None,
            },
        )
        if final_state.get("inline_eval_score") is not None:
            trace.score(name="inline_faithfulness", value=final_state["inline_eval_score"])
        _langfuse_client.flush()
 
    if answer:
        store_cached_response(query, answer)
 
    return final_state