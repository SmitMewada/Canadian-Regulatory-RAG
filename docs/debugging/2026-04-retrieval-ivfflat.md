# Retrieval Bug: IVFFLAT Index Truncating Vector Search Results

**Date:** April 2026
**Status:** Resolved
**Affected component:** `src/pipeline/nodes/retrieve.py`, `scripts/init.sql`

## Symptom

Vector search inside `retrieve_node` returned only 4 chunks for
unfiltered queries, despite `LIMIT 20` and a corpus of 398 chunks.
Filtered queries (e.g., `document_filter='pipeda'`) correctly
returned 20 chunks.

The returned `RetrievedChunk.score` values were flat at ~0.0164,
~0.0161, ~0.0159 across all queries, which initially looked like
a similarity calculation bug.

## Root Cause

Two distinct issues were conflated:

**1. The flat scores were RRF scores, not cosine similarity.**
`rrf_fusion()` uses the formula `1 / (k + rank)` with `k = 60`:
- rank 1: 1/61 = 0.01639
- rank 2: 1/62 = 0.01613
- rank 3: 1/63 = 0.01587

Raw cosine similarity was being computed correctly in SQL but
overwritten by the RRF score in `RetrievedChunk.score`.

**2. IVFFLAT index returned ~4 rows per query.**
The schema used `WITH (lists = 100)` on a 398-chunk corpus.
With pgvector's default `ivfflat.probes = 1`, queries search
exactly one inverted list. Average list size: 398/100 ≈ 4 chunks.
That's why `LIMIT 20` returned 4 rows.

Filtered queries escaped this because the planner switched to
sequential scan, bypassing the index entirely.

## Ruled Out

The following were investigated and confirmed **not** the cause:

- `RealDictCursor` interference with vector casting
- `%s::vector` parameter mangling by psycopg2
- Python list vs numpy array serialization
- Stale `__pycache__` or import shadowing
- Wrong `DATABASE_URL` in the module
- Different embedding models between ingestion and retrieval

## Fix

Dropped both `ivfflat` indexes. At ~400 chunks, exact sequential
scan over 384-dim vectors runs in a few ms and is correct by
construction. No ANN index is justified until the corpus grows
past ~10K chunks, at which point HNSW (not IVFFLAT) is the
recommended upgrade — see `scripts/init.sql`.

## Verification

Before fix:

[retrieve] vector=4 bm25=0   # unfiltered query — wrong
[retrieve] vector=20 bm25=0  # filtered query (seq scan bypass)


After fix:

[retrieve] vector=20 bm25=0  # unfiltered query — resolved
[retrieve] vector=20 bm25=0  # filtered query

Raw cosine scores for top result: 0.69–0.80 range, confirming
embedding model and vector cast are working correctly.

## Why Not SET LOCAL ivfflat.probes = 100?

The agent that validated this bug recommended `probes = 100` as
the fix. We disagreed and dropped the index instead.

`probes = 100` on `lists = 100` is just exact search with
bookkeeping overhead. It also requires every code path to remember
to set it — a future maintenance trap. Dropping the index is
cleaner, faster, and correct-by-construction at this corpus size.

## Open Issues

- BM25 returns 0 for all regulatory acronyms (PIPEDA, OSFI, E-23).
  PostgreSQL English dictionary drops unknown tokens. Vector search
  carries the full weight for acronym queries. Fix in production:
  Elasticsearch with custom analyzers.
- `RetrievedChunk.score` conflated RRF with similarity — refactored
  to expose `vector_score`, `bm25_score`, and `score` separately.