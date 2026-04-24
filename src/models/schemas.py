from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# --- Document Models ---

class DocumentRecord(BaseModel):
    """Represents a regulatory document in the corpus."""
    id: str                          # e.g., "osfi_e23"
    title: str
    source_url: str
    file_path: str
    sha256_hash: str
    date_accessed: datetime
    total_pages: Optional[int] = None


class ChunkRecord(BaseModel):
    """A single text chunk ready for embedding and storage."""
    document_id: str
    chunk_index: int
    chunk_text: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    metadata: dict = {}


class ChunkWithEmbedding(ChunkRecord):
    """ChunkRecord with embedding vector attached."""
    embedding: List[float]


# --- Query / Retrieval Models ---

class RetrievedChunk(BaseModel):
    """A chunk returned from hybrid search."""
    id: int
    chunk_text: str
    document_id: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    score: float                          # RRF combined score
    rerank_score: Optional[float] = None


# --- Response Models ---

class RegulatoryAnswer(BaseModel):
    """Final structured answer returned to the user."""
    answer: str
    cited_sources: List[str]              # ["E-23 Section 4.2", "PIPEDA s.7"]
    confidence: str                       # "high" / "medium" / "low"
    disclaimer: Optional[str] = None
    requires_human_review: bool = False


# --- Evaluation Models ---

class EvalTestCase(BaseModel):
    """A single evaluation test case."""
    id: str                               # e.g., "e23_001"
    question: str
    expected_answer: str
    source_document: str
    relevant_sections: List[str]
    category: str                         # factual / cross_doc / out_of_scope / guardrail
    difficulty: str                       # easy / medium / hard


# --- LangGraph State ---

from typing import TypedDict, Annotated
import operator

class GraphState(TypedDict):
    """State passed between LangGraph nodes."""
    original_query: str
    rewritten_query: str
    document_filter: Optional[str]
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]
    generated_answer: Optional[RegulatoryAnswer]
    inline_eval_score: Optional[float]
    citation_valid: Optional[bool]
    retry_count: int
    langfuse_trace_id: Optional[str]
    cache_hit: bool
    
class RetrievedChunk(BaseModel):
    id: int
    chunk_text: str
    document_id: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None  # add this
    score: float  