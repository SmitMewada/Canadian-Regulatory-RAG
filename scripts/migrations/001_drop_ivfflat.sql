-- Migration 001: drop ivfflat indexes
-- Context: ivfflat with lists=100 on a ~400-chunk corpus returned only
-- ~4 rows per vector query because probes=1 searches a single small list.
-- Exact sequential scan is fast and correct at this corpus size.
-- See init.sql for HNSW migration path when corpus grows.

DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_cache_embedding;