# src/pipeline/nodes/retrieve.py
import os
import psycopg2
import psycopg2.extras

from sentence_transformers import SentenceTransformer
from src.models.schemas import GraphState, RetrievedChunk
from langfuse import Langfuse

from dotenv import load_dotenv
load_dotenv()

print("DB:", os.getenv("DATABASE_URL")) 


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))

embedder = SentenceTransformer(EMBEDDING_MODEL)

_langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
)

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def embed_query(query: str) -> list[float]:
    # Return list — matches your prototype that worked
    return embedder.encode(query, normalize_embeddings=True).tolist()

def vector_search(cur, query_vec: list, document_filter: str, limit: int):
    filter_clause = f"AND document_id = '{document_filter}'" if document_filter else ""
    cur.execute(f"""
        SELECT id, chunk_text, document_id, section_heading, page_number,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE true {filter_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_vec, query_vec, limit))
    return cur.fetchall()

def bm25_search(cur, query: str, document_filter: str, limit: int):
    filter_clause = f"AND document_id = '{document_filter}'" if document_filter else ""
    cur.execute(f"""
        SELECT id, chunk_text, document_id, section_heading, page_number,
               ts_rank(search_vector, plainto_tsquery('english', %s)) AS score
        FROM chunks
        WHERE search_vector @@ plainto_tsquery('english', %s)
        {filter_clause}
        ORDER BY score DESC
        LIMIT %s
    """, (query, query, limit))
    return cur.fetchall()

def rrf_fusion(vector_rows, bm25_rows, k: int = 60, limit: int = 10):
    scores = {}
    meta = {}

    for rank, row in enumerate(vector_rows, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        meta[cid] = row

    for rank, row in enumerate(bm25_rows, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid not in meta:
            meta[cid] = row

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(meta[cid], score) for cid, score in ranked[:limit]]

def retrieve_node(state: GraphState) -> GraphState:
    query = state["rewritten_query"] or state["original_query"]
    document_filter = state["document_filter"]
    query_vec = embed_query(query)
    
    print(f"[retrieve] query = {query!r}")
    print(f"[retrieve] document_filter = {document_filter!r}")
    print(f"[retrieve] RETRIEVAL_TOP_K = {RETRIEVAL_TOP_K}")
    print(f"[retrieve] vec_dim = {len(query_vec)}, first 3 = {query_vec[:3]}")
    
    trace_id = state.get("langfuse_trace_id")
    span = _langfuse.span(
        trace_id=trace_id,
        name="hybrid_retrieve",
        input={"query": query, "document_filter": document_filter},
    ) if trace_id else None

    
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            vector_rows = vector_search(cur, query_vec, document_filter, limit=RETRIEVAL_TOP_K * 2)
            bm25_rows = bm25_search(cur, query, document_filter, limit=RETRIEVAL_TOP_K * 2)
            print(f"[retrieve] vector={len(vector_rows)} bm25={len(bm25_rows)}")
            if vector_rows:
                print(f"[retrieve] first vector result: doc={vector_rows[0]['document_id']}, raw_score={vector_rows[0]['score']}")
    finally:
        conn.close()

    fused = rrf_fusion(vector_rows, bm25_rows, limit=RETRIEVAL_TOP_K)

    chunks = [
        RetrievedChunk(
            id=row["id"],
            chunk_text=row["chunk_text"],
            document_id=row["document_id"],
            section_heading=row.get("section_heading"),
            page_number=row.get("page_number"),
            score=float(rrf_score),
        )
        for row, rrf_score in fused
    ]
    
    if span:
        top_rrf = chunks[0].score if chunks else 0.0
        span.end(
            output={"chunks_retrieved": len(chunks), "top_rrf_score": round(top_rrf, 4)},
            metadata={
                "vector_hits": len(vector_rows),
                "bm25_hits": len(bm25_rows),
                "document_filter": document_filter,
            },
        )
        _langfuse.score(
            trace_id=trace_id,
            name="chunks_retrieved",
            value=len(chunks),
        )
        _langfuse.score(
            trace_id=trace_id,
            name="top_rrf_score",
            value=round(top_rrf, 4),
        )

    return {**state, "retrieved_chunks": chunks}