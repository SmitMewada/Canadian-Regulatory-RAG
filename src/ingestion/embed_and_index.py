import json
import os
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.models.schemas import DocumentRecord, ChunkRecord, ChunkWithEmbedding

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def get_db_connection():
    """Get a PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def load_embedding_model() -> SentenceTransformer:
    """
    Load BGE embedding model locally.
    First run downloads ~130MB. Subsequent runs use cache.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(chunks: list[ChunkRecord], model: SentenceTransformer) -> list[ChunkWithEmbedding]:
    """
    Generate embeddings for all chunks in batches.
    BGE models work best with a query prefix for retrieval.
    For documents we use no prefix (plain text).
    """
    print(f"Embedding {len(chunks)} chunks...")

    texts = [chunk.chunk_text for chunk in chunks]

    # Batch embedding — more efficient than one at a time
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # Important for cosine similarity
    )

    chunks_with_embeddings = []
    for chunk, embedding in zip(chunks, embeddings):
        chunks_with_embeddings.append(
            ChunkWithEmbedding(
                **chunk.model_dump(),
                embedding=embedding.tolist(),
            )
        )

    return chunks_with_embeddings


def upsert_documents(records: list[DocumentRecord], conn) -> None:
    """Insert or update document records in the documents table."""
    cursor = conn.cursor()

    for record in records:
        cursor.execute("""
            INSERT INTO documents (id, title, source_url, file_path, sha256_hash, date_accessed, total_pages)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                sha256_hash = EXCLUDED.sha256_hash,
                date_accessed = EXCLUDED.date_accessed,
                updated_at = NOW()
        """, (
            record.id,
            record.title,
            record.source_url,
            record.file_path,
            record.sha256_hash,
            record.date_accessed,
            record.total_pages,
        ))

    conn.commit()
    cursor.close()
    print(f"Upserted {len(records)} document records")


def upsert_chunks(chunks: list[ChunkWithEmbedding], conn) -> None:
    """
    Insert chunks with embeddings into PostgreSQL.
    Clears existing chunks for each document first to avoid duplicates.
    """
    cursor = conn.cursor()

    # Get unique document IDs in this batch
    doc_ids = list(set(c.document_id for c in chunks))

    # Clear existing chunks for these documents
    for doc_id in doc_ids:
        cursor.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))

    print(f"Inserting {len(chunks)} chunks into PostgreSQL...")

    for i, chunk in enumerate(chunks):
        cursor.execute("""
            INSERT INTO chunks (
                document_id, chunk_index, chunk_text,
                section_heading, page_number, embedding, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            chunk.document_id,
            chunk.chunk_index,
            chunk.chunk_text,
            chunk.section_heading,
            chunk.page_number,
            chunk.embedding,  # pgvector accepts Python lists
            json.dumps(chunk.metadata),
        ))

        if (i + 1) % 50 == 0:
            print(f"  Inserted {i + 1}/{len(chunks)} chunks...")

    conn.commit()
    cursor.close()
    print(f"Done. {len(chunks)} chunks stored.")


def verify_index(conn) -> None:
    """Quick verification that chunks are in the DB correctly."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM chunks")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT document_id, COUNT(*)
        FROM chunks
        GROUP BY document_id
        ORDER BY document_id
    """)
    per_doc = cursor.fetchall()

    cursor.close()

    print(f"\nVerification — Total chunks in DB: {total}")
    print("Per document:")
    for doc_id, count in per_doc:
        print(f"  {doc_id}: {count} chunks")


def run_pipeline() -> None:
    """Full ingestion pipeline: manifest → extract → chunk → embed → store."""
    from src.ingestion.extract import extract_all
    from src.ingestion.chunk import chunk_pages

    # Load manifest
    manifest = json.loads(Path("data/manifest.json").read_text())
    records = [DocumentRecord(**v) for v in manifest.values()]

    # Extract text
    pages = extract_all(records)

    # Chunk
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")

    # Load embedding model
    model = load_embedding_model()

    # Embed
    chunks_with_embeddings = embed_chunks(chunks, model)

    # Store in DB
    conn = get_db_connection()
    try:
        upsert_documents(records, conn)
        upsert_chunks(chunks_with_embeddings, conn)
        verify_index(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()