# src/pipeline/cache.py
import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from src.models.schemas import RegulatoryAnswer

load_dotenv()

CACHE_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.98"))

from src.pipeline.nodes.retrieve import embedder

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def embed_to_str(query: str) -> str:
    """Embed query and return as pgvector-compatible string."""
    embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    return "[" + ",".join(str(x) for x in embedding) + "]"

def get_cached_response(query: str) -> RegulatoryAnswer | None:
    embedding_str = embed_to_str(query)

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT response_json,
                       1 - (query_embedding <=> %s::vector) AS similarity
                FROM query_cache
                ORDER BY query_embedding <=> %s::vector
                LIMIT 1
            """, (embedding_str, embedding_str))
            row = cur.fetchone()

            if row and row["similarity"] >= CACHE_THRESHOLD:
                print(f"  [cache] HIT — similarity={row['similarity']:.4f}")
                cur.execute("""
                    UPDATE query_cache SET hit_count = hit_count + 1
                    WHERE query_text = (
                        SELECT query_text FROM query_cache
                        ORDER BY query_embedding <=> %s::vector
                        LIMIT 1
                    )
                """, (embedding_str,))
                conn.commit()
                data = row["response_json"]
                if isinstance(data, str):
                    data = json.loads(data)
                return RegulatoryAnswer(**data)

            sim = f"{row['similarity']:.4f}" if row else "N/A"
            print(f"  [cache] MISS — similarity={sim}")
            return None
    finally:
        conn.close()

def store_cached_response(query: str, answer: RegulatoryAnswer) -> None:
    if answer.confidence == "low":
        print(f"  [cache] skipping store — low confidence")
        return

    embedding_str = embed_to_str(query)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_cache (query_text, query_embedding, response_json)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT DO NOTHING
            """, (
                query,
                embedding_str,
                json.dumps(answer.model_dump()),
            ))
            conn.commit()
            print(f"  [cache] stored '{query[:50]}'")
    finally:
        conn.close()