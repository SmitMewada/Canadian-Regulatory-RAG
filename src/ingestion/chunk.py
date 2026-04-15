from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.schemas import ChunkRecord


def create_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter.
    
    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraphs first, then sentences, then words
    - Preserves natural text boundaries better than fixed-size splitting
    - Important for regulatory docs where sentence context matters
    
    Why chunk_size=1000, overlap=200?
    - 1000 tokens fits comfortably in LLM context with room for answer
    - 200 token overlap prevents cutting mid-concept at boundaries
    - Your prototype logs confirmed chunk boundaries affect retrieval quality
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[ChunkRecord]:
    """
    Split extracted pages into chunks.
    
    Each page dict must have:
        document_id, page_number, text, section_heading
    
    Returns list of ChunkRecord objects ready for embedding.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    chunks = []
    chunk_index = 0

    for page in pages:
        # Split this page's text into chunks
        texts = splitter.split_text(page["text"])

        for text in texts:
            # Skip very short chunks — not useful for retrieval
            if len(text.strip()) < 50:
                continue

            chunk = ChunkRecord(
                document_id=page["document_id"],
                chunk_index=chunk_index,
                chunk_text=text.strip(),
                section_heading=page.get("section_heading"),
                page_number=page.get("page_number"),
                metadata={
                    "source_document": page["document_id"],
                    "page_number": page.get("page_number"),
                    "section_heading": page.get("section_heading"),
                },
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.models.schemas import DocumentRecord
    from src.ingestion.extract import extract_all

    # Load manifest
    manifest = json.loads(Path("data/manifest.json").read_text())
    records = [DocumentRecord(**v) for v in manifest.values()]

    # Extract pages
    pages = extract_all(records)

    # Chunk
    chunks = chunk_pages(pages)

    # Stats
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c.chunk_text) for c in chunks) // len(chunks)} chars")

    # Show distribution per document
    from collections import Counter
    doc_counts = Counter(c.document_id for c in chunks)
    print("\nChunks per document:")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"  {doc_id}: {count} chunks")

    # Sample chunk
    print("\n--- Sample Chunk ---")
    sample = chunks[10]
    print(f"Document: {sample.document_id}")
    print(f"Page: {sample.page_number}")
    print(f"Heading: {sample.section_heading}")
    print(f"Length: {len(sample.chunk_text)} chars")
    print(f"Text: {sample.chunk_text[:400]}")