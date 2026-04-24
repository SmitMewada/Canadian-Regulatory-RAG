# src/pipeline/nodes/rerank.py
import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from src.models.schemas import GraphState, RetrievedChunk

load_dotenv()

RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Load once at module level — expensive to reload per query
reranker = CrossEncoder(RERANK_MODEL)

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

    return {**state, "reranked_chunks": reranked}