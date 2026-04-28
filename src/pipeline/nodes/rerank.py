# src/pipeline/nodes/rerank.py
import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from src.models.schemas import GraphState, RetrievedChunk
from langfuse import Langfuse

load_dotenv()

RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Load once at module level — expensive to reload per query
reranker = CrossEncoder(RERANK_MODEL)

_langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
)

def rerank_node(state: GraphState) -> GraphState:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Cross-encoder reads query + chunk together — much more precise
    than bi-encoder similarity used in vector search.
    Reduces top_k=10 → top_k=5.
    """
    chunks = state["retrieved_chunks"]
    query = state["rewritten_query"] or state["original_query"]

    if not chunks:
        print("  [rerank] no chunks to rerank")
        return {**state, "reranked_chunks": []}
    
    trace_id = state.get("langfuse_trace_id")
    span = _langfuse.span(
        trace_id=trace_id,
        name="rerank",
        input={"query": query, "chunks_in": len(chunks)},
    ) if trace_id else None

    # Cross-encoder scores query against each chunk together
    pairs = [(query, chunk.chunk_text) for chunk in chunks]
    scores = reranker.predict(pairs)

    print(f"  [rerank] scoring {len(chunks)} chunks → keeping top {RERANK_TOP_K}")

    # Attach rerank score to each chunk
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top_k, update rerank_score field
    reranked = []
    for chunk, score in scored[:RERANK_TOP_K]:
        reranked.append(RetrievedChunk(
            id=chunk.id,
            chunk_text=chunk.chunk_text,
            document_id=chunk.document_id,
            section_heading=chunk.section_heading,
            page_number=chunk.page_number,
            vector_score=chunk.vector_score,
            bm25_score=chunk.bm25_score,
            score=chunk.score,           # keep original RRF score
            rerank_score=float(score),   # add rerank score
        ))

    # Log score improvement
    print(f"  [rerank] top 3 after rerank:")
    for i, c in enumerate(reranked[:3]):
        print(f"    [{i+1}] doc={c.document_id} rerank={c.rerank_score:.4f} rrf={c.score:.4f}")

    if span:
        top_score = reranked[0].rerank_score if reranked else 0.0
        bottom_score = reranked[-1].rerank_score if reranked else 0.0
        span.end(
            output={"chunks_out": len(reranked), "top_rerank_score": round(top_score, 4)},
            metadata={
                "bottom_rerank_score": round(bottom_score, 4),
                "score_spread": round(top_score - bottom_score, 4),
                "chunks_in": len(chunks),
                "chunks_out": len(reranked),
            },
        )
        _langfuse.score(trace_id=trace_id, name="rerank_top_score", value=round(top_score, 4))
        _langfuse.score(trace_id=trace_id, name="rerank_score_spread", value=round(top_score - bottom_score, 4))

    return {**state, "reranked_chunks": reranked}