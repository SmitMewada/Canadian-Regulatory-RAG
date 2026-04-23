# 🍁 Canadian Financial Regulatory RAG

> A production-grade Retrieval-Augmented Generation system for querying Canadian financial regulatory documents — with measurable reliability, full observability, and a CI/CD quality gate that blocks deployment on quality regression.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![Dagster](https://img.shields.io/badge/Dagster-1.7+-orange.svg)](https://dagster.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Eval Gate](https://img.shields.io/badge/CI-Eval%20Gate-critical.svg)](.github/workflows/eval-gate.yml)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What Makes This Different](#what-makes-this-different)
- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Document Corpus](#document-corpus)
- [Evaluation Results](#evaluation-results)
- [Cost Analysis](#cost-analysis)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Regulatory Context](#regulatory-context)
- [Limitations and Future Work](#limitations-and-future-work)
- [Lessons Learned](#lessons-learned)

---

## Problem Statement

Compliance teams at Canadian financial institutions spend significant time manually cross-referencing regulatory documents — OSFI guidelines, federal privacy law (PIPEDA), and Treasury Board AI directives. These documents are dense, interconnected, and updated periodically. A single question like *"What does OSFI E-23 require before deploying a credit-scoring model, and how does that interact with PIPEDA consent obligations?"* may require reading across three separate sources.

This system answers those questions in under 2 seconds with traceable, cited, hallucination-controlled answers — and provides evaluation metrics to prove it.



---

## What Makes This Different

Most RAG portfolio projects stop at "it retrieves and generates." This system treats the **evaluation pipeline and observability layer as first-class components**, not afterthoughts.

| Capability | Typical RAG Demo | This System |
|---|---|---|
| Retrieval | Vector search only | Hybrid BM25 + vector (Reciprocal Rank Fusion) |
| Answer quality | Vibes | RAGAS faithfulness ≥ 0.80, hallucination rate ≤ 0.10 |
| Observability | None | Langfuse traces: per-node latency, cost, token counts |
| CI/CD quality gate | None | GitHub Actions blocks merge on metric regression |
| Document updates | Manual re-run | Dagster sensor detects hash changes, auto re-ingests |
| Repeated queries | Full pipeline every time | Semantic cache (cosine similarity threshold) |
| Out-of-scope handling | Hallucination | "I don't know" accuracy ≥ 0.85, tested in CI |
| Financial advice guardrail | None | Disclaimer triggered on 100% of advice-seeking queries |
| Runtime safety | None | Inline faithfulness eval — flags low-confidence answers |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SEMANTIC CACHE CHECK                           │
│         cosine similarity ≥ 0.95 → return cached response       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ cache miss
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               LANGGRAPH QUERY PIPELINE                           │
│                                                                  │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │ Query Rewrite│───▶│  Hybrid Retrieve  │───▶│   Rerank     │  │
│  │ (preserve    │    │  BM25 + pgvector  │    │  (Cohere)    │  │
│  │  reg. refs)  │    │  RRF fusion top10 │    │  top10→top5  │  │
│  └──────────────┘    └───────────────────┘    └──────┬───────┘  │
│                                                       │          │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────▼───────┐  │
│  │ Format Output│◀───│  Inline Eval      │◀───│   Generate   │  │
│  │ + Disclaimer │    │  (faithfulness    │    │  (Instructor │  │
│  │   Guardrail  │    │   score 1–5)      │    │   + Pydantic)│  │
│  └──────────────┘    └───────────────────┘    └──────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ every node traced
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LANGFUSE OBSERVABILITY                           │
│    latency │ token cost │ retrieval scores │ eval scores        │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               FASTAPI RESPONSE (RegulatoryAnswer)                │
│   answer │ cited_sources │ confidence │ disclaimer │ trace_id   │
└─────────────────────────────────────────────────────────────────┘
```

### Ingestion Pipeline (Dagster)

```
Regulatory PDFs/HTML
        │
        ▼
┌──────────────────┐
│  raw_documents   │  Download + SHA-256 hash manifest
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  extracted_text  │  PyMuPDF text extraction + cleaning
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│chunked_documents │  RecursiveCharacterTextSplitter
│                  │  chunk_size=1000, overlap=200
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│db_docs │ │db_chunks │  pgvector + ts_vector (BM25)
└────────┘ └──────────┘
         ▲
         │
┌────────┴──────────────┐
│ document_update_sensor │  Daily hash check → triggers re-ingestion
└───────────────────────┘
```

### System Layers

| Layer | Component | Responsibility |
|---|---|---|
| Orchestration | Dagster | Asset-based ingestion pipeline with lineage. Sensor detects document updates via SHA-256 hash comparison. |
| Storage | PostgreSQL 15 + pgvector | Hybrid queries (BM25 full-text + cosine similarity) in one database. Joins vector results with relational metadata. |
| Query Pipeline | LangGraph | State machine: rewrite → retrieve → rerank → generate → eval → format. Retry on low faithfulness. |
| Observability | Langfuse v3 (self-hosted) | Per-node traces with latency, token cost, retrieval scores, and inline eval results. |
| Serving | FastAPI + Streamlit | Typed Pydantic responses. Minimal Streamlit demo frontend. |
| Evaluation | DeepEval + RAGAS + GitHub Actions | 40+ hand-curated test cases. CI gate blocks merge if metrics regress. |

---

## Key Design Decisions

### 1. Why PostgreSQL + pgvector instead of a dedicated vector database?

Toronto banks run PostgreSQL. Using pgvector means hybrid search (BM25 + cosine similarity), metadata pre-filtering, and vector search all happen in one database with standard SQL joins. Adding a dedicated vector DB (Qdrant, Pinecone) would introduce infrastructure complexity with no benefit for this corpus size. In production at scale, a dedicated ANN index would be warranted — documented in Limitations.

### 2. Why Reciprocal Rank Fusion (RRF) for hybrid search?

RRF combines BM25 and vector rankings without requiring normalization of scores across different scales. The formula `1 / (k + rank)` with k=60 is well-studied and outperforms naive score averaging. Regulatory documents benefit significantly from BM25 because compliance queries often contain specific clause identifiers (e.g., "E-23 Section 4.2") that semantic search alone would underweight.

**Lesson learned from prototyping:** Dense retrieval is "topic-aware" but not "answer-aware." BM25 is "word-aware" but not "context-aware." Hybrid search improves recall, not precision — which is why the Cohere reranker (a cross-encoder) is a required second stage.

### 3. Why chunk_size=1000 tokens with 200 overlap?

Canadian regulatory documents have dense, self-referential structure. Smaller chunks (256–512 tokens) lose the context needed to answer multi-part questions. Larger chunks (2000+ tokens) dilute retrieval signal. 1000 tokens captures a full subsection while remaining focused. The 200-token overlap ensures clause boundaries don't split critical context. This was validated by inspecting actual retrieval failures during prototyping.

### 4. Why hand-curated evaluation dataset?

LLM-generated evaluation sets are circular — you are evaluating an LLM using another LLM's judgments about the same content. The 40+ test cases in `eval/eval_dataset.json` were written by hand to cover: factual extraction, cross-document reasoning, out-of-scope detection, advice-seeking guardrails, and exact clause reference (the hardest case for hybrid search). This cannot be automated.

### 5. Why Langfuse v3 (self-hosted) instead of a managed observability service?

Canadian financial services data governance requirements make sending query data to third-party servers problematic. Self-hosting Langfuse keeps all traces on-premises. The MIT license allows production deployment without licensing cost. **Critical version note:** Langfuse SDK v2 must be paired with Langfuse server v3. Never use `latest` tags — pin both explicitly. SDK v4 uses OpenTelemetry ingestion protocol incompatible with server v3.

### 6. Why LiteLLM for LLM calls?

LiteLLM provides a unified interface across OpenAI, Anthropic, and local models. Switching from `gpt-4o-mini` to `claude-3-haiku` for cost optimization requires only a single environment variable change. This is a production-credible pattern used in real financial services AI deployments.

### 7. Why Instructor + Pydantic for structured output?

**Lesson learned from prototyping:** Prompting alone does not guarantee structured output. The model returned `"null"` as a string, confidence values of 1.0 regardless of quality, and missing fields. Instructor wraps the LLM call with retry logic and schema validation — if the model returns invalid JSON, Instructor retries automatically. Pydantic enforces the contract at the type level.

### 8. Why three layers of hallucination defense?

- **Retrieval grounding:** The generation prompt explicitly instructs the model to use only the provided context.
- **Inline faithfulness eval:** An LLM-as-judge scores each answer 1–5 at runtime. Score < 3 triggers retry with expanded retrieval.
- **Offline CI gate:** 40+ test cases with a hallucination rate threshold ≤ 0.10 run on every pull request. Merge is blocked on regression.

No single layer is sufficient. All three are required for production-grade reliability.

---

## Document Corpus

Five carefully selected Canadian regulatory documents. A small, diverse corpus creates harder retrieval challenges than a large flat corpus.

| Document | Regulatory Relevance | Source |
|---|---|---|
| OSFI Guideline E-23 (Model Risk Management, eff. May 2027) | The primary AI/ML governance regulation for federally regulated financial institutions | osfi-bsif.gc.ca |
| Treasury Board Directive on Automated Decision-Making | Most enforceable federal AI governance instrument. Requires Algorithmic Impact Assessments. | tbs-sct.canada.ca |
| PIPEDA (key sections) | Federal privacy law governing personal data in commercial AI systems | priv.gc.ca |
| OSFI Guideline B-13 (Technology & Cyber Risk) | Complements E-23 for infrastructure and technology risk | osfi-bsif.gc.ca |
| Government of Canada Algorithmic Impact Assessment | The actual framework used to score AI system risk levels under the TB Directive | github.com/canada-ca |

All documents are publicly available from official government sources. Source URLs are in `data/manifest.json`.

---

## Evaluation Results

> Results below reflect the evaluation run on `main` at the time of writing. Current results are always available as GitHub Actions artifacts.

### Metric Scorecard

| Metric | Tool | Threshold | Result | Status |
|---|---|---|---|---|
| Faithfulness | RAGAS | ≥ 0.80 | 0.86 | ✅ PASS |
| Answer Relevancy | RAGAS | ≥ 0.75 | 0.81 | ✅ PASS |
| Hallucination Rate | DeepEval | ≤ 0.10 | 0.07 | ✅ PASS |
| Context Precision | RAGAS | ≥ 0.70 | 0.74 | ✅ PASS |
| Context Recall | RAGAS | ≥ 0.65 | 0.71 | ✅ PASS |
| "I Don't Know" Accuracy | Custom | ≥ 0.85 | 0.90 | ✅ PASS |
| Disclaimer Trigger Rate | Custom | 100% | 100% | ✅ PASS |
| Citation Validity | Custom | ≥ 0.90 | 0.93 | ✅ PASS |
| End-to-End Latency | Langfuse | < 2.0s | 1.4s avg | ✅ PASS |
| Cost per Query | LiteLLM | < $0.01 | $0.006 | ✅ PASS |

### Evaluation Dataset Breakdown (40 test cases)

| Category | Count | Pass Rate |
|---|---|---|
| Factual extraction (single document) | 15 | 93% |
| Cross-document reasoning | 8 | 75% |
| Out-of-scope ("I don't know") | 7 | 100% |
| Guardrail (advice-seeking, disclaimer) | 5 | 100% |
| Exact clause reference | 5 | 80% |

**Where the system struggles:** Cross-document reasoning (75% pass rate) is the hardest category. Questions like *"Which Canadian regulations require an impact assessment before deploying AI?"* require synthesizing context from both the TB Directive and OSFI E-23. Retrieval precision drops when the query does not contain document-specific keywords. Ongoing work: expand the hybrid search reranking step to include cross-document re-scoring.

---

## Cost Analysis

Running costs at the time of writing, using `gpt-4o-mini` for generation and `text-embedding-3-small` for embeddings.

| Component | Cost per Query | Notes |
|---|---|---|
| Embedding (query) | ~$0.00002 | text-embedding-3-small, 1 query embedding |
| LLM generation | ~$0.004 | gpt-4o-mini, ~800 input + ~300 output tokens |
| Reranking | ~$0.001 | Cohere Rerank API, 10 documents |
| Inline eval LLM call | ~$0.001 | gpt-4o-mini, ~500 input + ~10 output tokens |
| **Total (cache miss)** | **~$0.006** | |
| **Total (cache hit)** | **~$0.00002** | Only query embedding cost |

**At 1,000 queries/day with 30% cache hit rate:** ~$4.20/day, ~$127/month. Switching to `claude-3-haiku` via LiteLLM reduces generation cost by approximately 40% with minimal quality impact. Model choice is a single environment variable.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (for embeddings and generation)
- Cohere API key (free tier — [register here](https://cohere.com/))

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/canadian-regulatory-rag.git
cd canadian-regulatory-rag

# Copy the environment template
cp .env.example .env

# Fill in your API keys
nano .env
```

Your `.env` file needs:
```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...      # Created after first Langfuse boot (step 3)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_NEXTAUTH_SECRET=your-random-32-char-secret
LANGFUSE_SALT=your-random-32-char-salt
LITELLM_MODEL=gpt-4o-mini
```

### 2. Start all services

```bash
docker compose up -d
```

This starts PostgreSQL (with pgvector), Langfuse (web + worker), the FastAPI backend, and the Streamlit frontend.

Wait for services to be healthy:
```bash
docker compose ps   # All services should show "healthy" or "running"
```

> ⚠️ **Version pinning note:** The docker-compose.yml pins `langfuse/langfuse:3` and `langfuse/langfuse-worker:3`. Never use `latest` — Langfuse v4 uses an incompatible ingestion protocol. The SDK is pinned to `langfuse==2.60.0` for the same reason.

### 3. Set up Langfuse

Open [http://localhost:3000](http://localhost:3000) and create an account. Create a project and copy the Public Key and Secret Key into your `.env` file, then restart the API container:

```bash
docker compose restart api
```

### 4. Run the ingestion pipeline

```bash
# Run the Dagster ingestion pipeline
docker compose exec api python -m src.ingestion.run_pipeline
```

This downloads all regulatory documents, extracts text, chunks, embeds, and loads into PostgreSQL. Expect ~5–10 minutes on first run.

You can also run and monitor the pipeline via the Dagster UI at [http://localhost:3001](http://localhost:3001).

### 5. Query the system

**Streamlit UI:** [http://localhost:8501](http://localhost:8501)

**REST API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What risk tiers does OSFI E-23 define for AI models?"}'
```

**Example response:**
```json
{
  "answer": "OSFI Guideline E-23 defines three risk tiers for models: Tier 1 (high impact), Tier 2 (moderate impact), and Tier 3 (low impact). The tier assignment determines the rigor of validation, documentation, and ongoing monitoring required. High-impact models — such as those used in credit adjudication or capital calculation — require independent model validation and board-level oversight.",
  "cited_sources": ["OSFI E-23, Section 3.1", "OSFI E-23, Section 4.2"],
  "confidence": "high",
  "disclaimer": null,
  "requires_human_review": false
}
```

### 6. Run the evaluation suite

```bash
python eval/run_evaluation.py --dataset eval/eval_dataset.json
```

Results are written to `eval/results/`. The CI gate thresholds are checked by:
```bash
python eval/check_gate.py \
  --min-faithfulness 0.8 \
  --max-hallucination 0.1 \
  --min-relevancy 0.75 \
  --min-idk-accuracy 0.85
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Main query endpoint. Returns `RegulatoryAnswer`. |
| `/query/{trace_id}` | GET | Retrieve past query with full Langfuse trace. |
| `/health` | GET | Health check — verifies all services are up. |
| `/metrics` | GET | Aggregate metrics: avg latency, cost, eval scores. |
| `/feedback` | POST | User feedback (thumbs up/down + note) stored in Langfuse. |

---

## Project Structure

```
canadian-regulatory-rag/
├── README.md                          # This file (design document)
├── docker-compose.yml                 # Full stack: postgres, langfuse, api, frontend
├── Dockerfile                         # API container
├── Dockerfile.frontend                # Streamlit container
├── pyproject.toml                     # Python 3.11+ dependencies
├── .env.example                       # Environment variable template
├── .github/
│   └── workflows/
│       └── eval-gate.yml              # CI quality gate — blocks merge on regression
├── scripts/
│   └── init.sql                       # PostgreSQL schema (pgvector + ts_vector)
├── data/
│   ├── raw/                           # Original PDFs and HTML files
│   ├── processed/                     # Cleaned chunks with metadata
│   └── manifest.json                  # Document registry with SHA-256 hashes + source URLs
├── src/
│   ├── ingestion/
│   │   ├── download.py                # Fetches documents, validates hashes
│   │   ├── extract.py                 # PyMuPDF text extraction + cleaning
│   │   ├── chunk.py                   # RecursiveCharacterTextSplitter logic
│   │   ├── embed_and_index.py         # Embedding generation + pgvector upsert
│   │   └── dagster_assets.py          # Dagster asset graph + document update sensor
│   ├── pipeline/
│   │   ├── graph.py                   # LangGraph state machine (core file)
│   │   ├── nodes/
│   │   │   ├── query_rewrite.py       # Rephrase query, preserve regulatory references
│   │   │   ├── retrieve.py            # Hybrid BM25 + vector (Reciprocal Rank Fusion)
│   │   │   ├── rerank.py              # Cohere Rerank API
│   │   │   ├── generate.py            # LLM generation via Instructor + LiteLLM
│   │   │   └── inline_eval.py         # Runtime faithfulness scoring
│   │   ├── prompts/
│   │   │   ├── rewrite.txt            # Query rewrite prompt
│   │   │   ├── generate.txt           # Generation prompt
│   │   │   └── inline_eval.txt        # Faithfulness evaluation prompt
│   │   ├── guardrails.py              # Disclaimer injection + citation validation
│   │   └── cache.py                   # Semantic cache (cosine similarity)
│   ├── models/
│   │   └── schemas.py                 # Pydantic: RegulatoryAnswer, RetrievedChunk, GraphState
│   ├── api/
│   │   └── main.py                    # FastAPI endpoints
│   └── frontend/
│       └── app.py                     # Streamlit UI
├── eval/
│   ├── eval_dataset.json              # 40+ hand-curated question-answer pairs
│   ├── run_evaluation.py              # DeepEval + RAGAS evaluation runner
│   ├── check_gate.py                  # CI threshold checker
│   ├── dagster_assets.py              # Eval assets (run as Dagster materializations)
│   └── results/                       # Eval output (gitignored, uploaded as CI artifact)
├── notebooks/
│   └── exploration.ipynb              # Prototype experiments
└── docs/
    ├── architecture.png               # Architecture diagram
    ├── design_decisions.md            # Extended design rationale
    └── eval_report.md                 # Detailed evaluation results with charts
```

---

## Regulatory Context

### Why these documents?

**OSFI Guideline E-23 (Model Risk Management)** is the single most consequential AI regulation for federally regulated financial institutions (FRFIs) in Canada. Effective May 2027, it requires model risk tiers, independent validation, ongoing monitoring, and board-level accountability for high-impact models. Any bank deploying an ML model for credit, fraud, AML, or capital calculation must comply.

**Treasury Board Directive on Automated Decision-Making** is the most enforceable federal AI governance instrument. It mandates Algorithmic Impact Assessments before deploying automated decision systems in government, and sets out requirements that are rapidly influencing private sector norms.

**PIPEDA** (Personal Information Protection and Electronic Documents Act) is the baseline federal privacy law. Every AI system that processes personal data of Canadians must comply. It governs consent, purpose limitation, and subject rights — all of which intersect with model training and inference.

**OSFI B-13** complements E-23 with requirements for technology and cyber risk management, including third-party model risk and cloud deployment standards.

**Algorithmic Impact Assessment Tool** is the actual GitHub-hosted framework used to score AI systems under the TB Directive. Including it in the corpus allows the system to answer questions about how to self-assess AI deployment risk.

### How regulation shaped design

- **Temperature = 0** on all LLM calls. Compliance answers must be deterministic and reproducible — regulatory citations cannot shift between runs.
- **Citation validation:** Every source cited in an answer must map to an actual retrieved chunk. Fabricated citations are a compliance failure, not just a quality issue.
- **`requires_human_review` flag:** Questions that ask for specific compliance determinations ("does our model comply with E-23?") set this flag to true. The system provides information; qualified professionals make determinations.
- **Financial advice disclaimer:** 100% trigger rate on advice-seeking queries is a hard requirement, not a metric to optimize.
- **Document version tracking:** Regulatory documents change. The Dagster sensor and SHA-256 hash comparison ensure the system re-ingests when source documents are updated — and re-evaluation runs automatically via CI.

---

## Limitations and Future Work

### Known Limitations

**Cross-document reasoning** is the weakest point. The RRF retrieval step retrieves the top-k chunks independently. When a question requires synthesizing two clauses from different documents, both must independently rank in the top-5 after reranking. A multi-hop retrieval strategy (retrieve → generate sub-questions → retrieve again) would improve this.

**Confidence scoring is not calibrated.** The `confidence` field in `RegulatoryAnswer` is derived from the inline faithfulness score, not from a calibrated probability. Do not treat it as a probability estimate.

**No streaming.** The current API returns the complete response synchronously. For latency-sensitive production deployments, FastAPI's `StreamingResponse` with LangGraph's streaming API would reduce time-to-first-token.

**Evaluation dataset is small.** 40–50 cases is sufficient to demonstrate the capability and catch regressions, but not large enough to claim statistical significance on metrics. Production deployment would require 500+ cases.

**No RBAC or multi-tenancy.** The API has no authentication. Production deployment requires OAuth 2.0 and query logging per user.

### Production Upgrade Path

| Component | Current | Production Upgrade |
|---|---|---|
| Orchestration | Docker Compose | Kubernetes (EKS/GKE) with Helm charts |
| Embedding model | OpenAI text-embedding-3-small | Fine-tuned embedding model on regulatory text |
| Infrastructure | Local | Terraform + AWS (RDS, ECS, ElastiCache) |
| Vector index | IVFFlat (pgvector) | HNSW for lower query latency at scale |
| Auth | None | OAuth 2.0 + JWT with per-user query logging |
| Eval dataset | 40 cases | 500+ cases with adversarial augmentation |
| Retrieval | Single-hop RRF | Multi-hop retrieval for cross-document reasoning |

---

## Lessons Learned

These are real lessons from the prototyping and build phases — not aspirational advice.

### Retrieval

- **Hybrid only improves recall, not precision.** BM25 + vector search returns better candidates; it does not guarantee the right answer ranks first. A cross-encoder reranker (Cohere Rerank or local BGE) is required as a second stage.
- **Chunk boundaries determine retrieval quality.** A chunk that splits a clause mid-sentence will never retrieve well. Time spent on chunking strategy pays off more than time spent tuning embedding parameters.
- **Metadata pre-filtering is a precision multiplier.** When a user references a specific document ("What does E-23 say..."), filtering retrieval to that document alone dramatically improves precision.

### Structured Output and Guardrails

- **Prompt ≠ guarantee.** Even with an explicit schema in the prompt, models returned `"null"` as a string, confidence values of 1.0 regardless of quality, and missing required fields. Instructor + Pydantic is the validation layer that enforces the contract.
- **Deterministic rules beat model behavior for safety.** The disclaimer trigger is a string-match rule, not an LLM judgment call. It fires 100% of the time because it has to.
- **LLM-reported confidence is not real confidence.** The model's stated confidence does not correlate with answer accuracy. Use retrieval scores and faithfulness eval as proxies — never surface the raw model confidence to users.

### Observability

- **SDK/server version pairing is strict.** Langfuse SDK v2 requires Langfuse server v3. SDK v4 uses OpenTelemetry ingestion protocol, which server v3 silently rejects. Auth succeeds, HTTP returns 200, but traces never appear in the UI. Always pin both explicitly; never use `latest`.
- **Langfuse v3 requires both containers.** `langfuse` (web/API) and `langfuse-worker` (event processor) are separate services. Without the worker, traces are accepted and acknowledged but never written to Clickhouse.
- **"Auth OK" does not mean "traces are arriving."** Test ingestion with a raw HTTP call to confirm the ingestion protocol is compatible before debugging the application layer.
- **Always call `flush()` before exit.** Langfuse buffers traces in a background thread. Scripts that exit before the thread finishes lose traces silently.

### Evaluation

- **Do not generate your eval set with an LLM.** The evaluation becomes circular — you are using one LLM to judge another LLM's answers about content that both LLMs have seen. Write the cases by hand from the source documents.
- **Out-of-scope handling is harder to evaluate than in-scope answers.** The system can correctly return "I don't know," but a naive evaluator gives that a low score for not answering. Evaluation logic must explicitly handle and reward correct abstention.

## Contributing

Contributions are welcome. If you find a bug or have a suggestion, open an issue. PRs that add test cases to `eval/eval_dataset.json` are especially encouraged — a larger, more diverse evaluation dataset directly improves system reliability.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built by Smit Mewada | March 2026*