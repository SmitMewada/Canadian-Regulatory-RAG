"""
eval/run_evaluation.py
Phase 4 — Offline Evaluation Suite

Runs all test cases from eval_dataset.json through the pipeline,
computes RAGAS + DeepEval metrics, writes results to eval/results/latest.json.

Usage:
    python eval/run_evaluation.py
    python eval/run_evaluation.py --dataset eval/eval_dataset.json
    python eval/run_evaluation.py --dry-run          # skip pipeline, use saved answers
    python eval/run_evaluation.py --category factual  # run one category only
    python eval/run_evaluation.py --id e23_001        # run a single test case

Environment variables required:
    XAI_API_KEY         — for pipeline generation (Grok)
    DATABASE_URL        — for pipeline retrieval (PostgreSQL)
    OPENAI_API_KEY      — for RAGAS judge model (gpt-4o-mini)
                          If not set, falls back to xAI judge (slower, self-consistency risk)

Architecture note:
    - Pipeline (Grok via xAI) generates answers
    - RAGAS judge (gpt-4o-mini via OpenAI) evaluates them
    - Using a DIFFERENT model as judge than as generator is intentional
    - Avoids the self-consistency bias documented in Phase 3 logs (LLM scored itself 5/5)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load env before any imports that read env vars
load_dotenv()

# ── RAGAS imports ──────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# ── DeepEval imports ───────────────────────────────────────────────────────────
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# ── LangChain LLM for RAGAS judge ─────────────────────────────────────────────
# IMPORTANT: This is the judge model, NOT the generation model.
# We use OpenAI gpt-4o-mini here even though the pipeline uses Grok.
# Reason: avoid self-consistency bias (same model judging its own outputs).
# If OPENAI_API_KEY is not set, we fall back to xAI — but note the bias risk.
def _build_ragas_judge():
    """
    Build the LLM and embedding wrappers RAGAS will use for evaluation.
    Separate from the pipeline's generation model by design.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        # Preferred path: OpenAI gpt-4o-mini as independent judge
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("[judge] Using OpenAI gpt-4o-mini as RAGAS judge (independent from pipeline)")
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    else:
        # Fallback: xAI via OpenAI-compatible client
        # WARNING: same provider as generation model — self-consistency bias risk
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        xai_key = os.getenv("XAI_API_KEY")
        assert xai_key, "Neither OPENAI_API_KEY nor XAI_API_KEY is set. Cannot build RAGAS judge."
        print("[judge] WARNING: Using xAI as RAGAS judge — same provider as pipeline generator.")
        print("[judge] Self-consistency bias risk: scores may be inflated for legitimate answers.")
        print("[judge] Set OPENAI_API_KEY to use an independent judge model.")
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="grok-3-fast",
                temperature=0,
                openai_api_key=xai_key,
                openai_api_base="https://api.x.ai/v1",
            )
        )
        # For embeddings fallback, use a local BGE model to avoid xAI dependency
        # This matches your ingestion pipeline's embedding model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )

    return llm, embeddings


# ── Pipeline import ────────────────────────────────────────────────────────────
# Import your actual pipeline. Adjust this path to match your project structure.
# We import lazily (inside functions) so --dry-run works without DB connection.
def _import_pipeline():
    """
    Import the LangGraph pipeline. Lazy import so --dry-run skips DB connection.
    Adjust the import path to match your src/ structure.
    """
    try:
        # Try the expected path from the spec's directory structure
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.pipeline.graph import run_pipeline  # noqa: F401
        return run_pipeline
    except ImportError as e:
        print(f"[ERROR] Could not import pipeline: {e}")
        print("Adjust the import path in _import_pipeline() to match your src/ structure.")
        print("Expected: src/pipeline/graph.py with a run_pipeline(query, document_filter) function")
        sys.exit(1)


# ── Run a single test case through the pipeline ────────────────────────────────
def run_test_case(pipeline_fn, test_case: dict) -> dict:
    """
    Run one test case through the pipeline.
    Returns a result dict with the pipeline's answer and retrieved contexts.

    The pipeline must return something we can extract:
        - answer text
        - retrieved chunk texts (for RAGAS context fields)
        - confidence
        - disclaimer (if triggered)
        - requires_human_review

    Adjust the extraction logic below if your GraphState keys differ.
    """
    question = test_case["question"]

    # Map source_document to document_filter
    # "cross_document" and "out_of_scope" → None (search all docs)
    # Any specific document ID → pass it as filter
    raw_source = test_case["source_document"]
    document_filter: Optional[str] = None
    if raw_source not in ("cross_document", "out_of_scope"):
        document_filter = raw_source

    start_time = time.time()

    try:
        # Call your pipeline
        # Expected signature: run_pipeline(query: str, document_filter: Optional[str]) -> GraphState
        # Adjust if your pipeline takes a different input format
        state = pipeline_fn(
            query=question,
            document_filter=document_filter,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract answer from GraphState
        # Based on your logs: state["generated_answer"] is a RegulatoryAnswer Pydantic model
        reg_answer = state.get("generated_answer")

        if reg_answer is None:
            return {
                "id": test_case["id"],
                "question": question,
                "answer": "Pipeline returned no answer.",
                "contexts": [],
                "confidence": "low",
                "disclaimer": None,
                "requires_human_review": False,
                "cited_sources": [],
                "latency_ms": latency_ms,
                "error": "generated_answer is None",
                "cache_hit": state.get("cache_hit", False),
            }

        # Extract retrieved chunk texts for RAGAS context fields
        # RAGAS needs the actual text strings, not citation labels
        reranked = state.get("reranked_chunks", [])
        if not reranked:
            reranked = state.get("retrieved_chunks", [])

        # Handle both dict and Pydantic model formats (your logs showed both)
        contexts = []
        for chunk in reranked:
            if hasattr(chunk, "chunk_text"):
                contexts.append(chunk.chunk_text)
            elif isinstance(chunk, dict):
                contexts.append(chunk.get("chunk_text", ""))

        # Extract RegulatoryAnswer fields
        # Handle both Pydantic model and dict (your logs showed psycopg2 sometimes returns dicts)
        if hasattr(reg_answer, "answer"):
            answer_text = reg_answer.answer
            confidence = reg_answer.confidence
            disclaimer = reg_answer.disclaimer
            requires_human_review = reg_answer.requires_human_review
            cited_sources = reg_answer.cited_sources
        else:
            answer_text = reg_answer.get("answer", "")
            confidence = reg_answer.get("confidence", "low")
            disclaimer = reg_answer.get("disclaimer")
            requires_human_review = reg_answer.get("requires_human_review", False)
            cited_sources = reg_answer.get("cited_sources", [])

        return {
            "id": test_case["id"],
            "question": question,
            "answer": answer_text,
            "contexts": contexts,
            "confidence": confidence,
            "disclaimer": disclaimer,
            "requires_human_review": requires_human_review,
            "cited_sources": cited_sources,
            "latency_ms": latency_ms,
            "error": None,
            "cache_hit": state.get("cache_hit", False),
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "id": test_case["id"],
            "question": question,
            "answer": "",
            "contexts": [],
            "confidence": "low",
            "disclaimer": None,
            "requires_human_review": False,
            "cited_sources": [],
            "latency_ms": latency_ms,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            "cache_hit": False,
        }


# ── Collect pipeline answers ───────────────────────────────────────────────────
def collect_answers(test_cases: list, dry_run: bool, saved_answers_path: Optional[str]) -> list:
    """
    Run all test cases through the pipeline (or load saved answers for --dry-run).
    Returns list of result dicts.
    """
    if dry_run:
        assert saved_answers_path and Path(saved_answers_path).exists(), (
            f"--dry-run requires a saved answers file. "
            f"Expected at: {saved_answers_path}\n"
            f"Run without --dry-run first to generate it."
        )
        print(f"[dry-run] Loading saved answers from {saved_answers_path}")
        with open(saved_answers_path) as f:
            data = json.load(f)
        return data["raw_answers"]

    pipeline_fn = _import_pipeline()
    results = []
    total = len(test_cases)

    print(f"\nRunning {total} test cases through pipeline...")
    print("─" * 60)

    for i, tc in enumerate(test_cases, 1):
        print(f"[{i:02d}/{total}] {tc['id']} — {tc['question'][:60]}...")
        result = run_test_case(pipeline_fn, tc)

        if result["error"]:
            print(f"         ✗ ERROR: {result['error'][:80]}")
        elif result["cache_hit"]:
            print(f"         ✓ {result['latency_ms']:.0f}ms (cache hit) | confidence={result['confidence']}")
        else:
            print(f"         ✓ {result['latency_ms']:.0f}ms | confidence={result['confidence']}")

        results.append(result)

        # Small delay to avoid rate limiting on the judge model
        if i < total:
            time.sleep(0.5)

    return results


# ── Custom metrics (not RAGAS or DeepEval) ────────────────────────────────────
def compute_custom_metrics(test_cases: list, answers: list) -> dict:
    """
    Compute metrics that RAGAS and DeepEval don't cover:
    - IDK accuracy: out_of_scope questions must return I-don't-know style answers
    - Disclaimer trigger rate: guardrail questions must have disclaimer set
    - Citation validity: cited sources must exist in the retrieved contexts
    - Error rate: pipeline crashes
    """
    tc_by_id = {tc["id"]: tc for tc in test_cases}
    ans_by_id = {a["id"]: a for a in answers}

    # ── IDK accuracy ──────────────────────────────────────────────────────────
    idk_phrases = [
        "i don't have sufficient information",
        "not included in this system",
        "not available in the",
        "outside the scope",
        "not in the corpus",
        "cannot answer",
        "no information",
    ]
    oos_cases = [tc for tc in test_cases if tc["category"] == "out_of_scope"]
    idk_correct = 0
    idk_details = []
    for tc in oos_cases:
        ans = ans_by_id.get(tc["id"], {})
        answer_text = ans.get("answer", "").lower()
        is_idk = any(phrase in answer_text for phrase in idk_phrases)
        # Also check confidence — out-of-scope should return low confidence
        is_low_confidence = ans.get("confidence", "") == "low"
        passed = is_idk or is_low_confidence
        if passed:
            idk_correct += 1
        idk_details.append({
            "id": tc["id"],
            "passed": passed,
            "is_idk_phrase": is_idk,
            "is_low_confidence": is_low_confidence,
            "answer_preview": answer_text[:100],
        })

    idk_accuracy = idk_correct / len(oos_cases) if oos_cases else 0.0

    # ── Disclaimer trigger rate ───────────────────────────────────────────────
    guardrail_cases = [tc for tc in test_cases if tc["category"] == "guardrail"]
    disclaimer_triggered = 0
    disclaimer_details = []
    for tc in guardrail_cases:
        ans = ans_by_id.get(tc["id"], {})
        has_disclaimer = bool(ans.get("disclaimer"))
        has_review_flag = ans.get("requires_human_review", False)
        passed = has_disclaimer and has_review_flag
        if passed:
            disclaimer_triggered += 1
        disclaimer_details.append({
            "id": tc["id"],
            "passed": passed,
            "has_disclaimer": has_disclaimer,
            "requires_human_review": has_review_flag,
        })

    disclaimer_rate = disclaimer_triggered / len(guardrail_cases) if guardrail_cases else 0.0

    # ── Citation validity ─────────────────────────────────────────────────────
    # Each cited source should correspond to something in retrieved contexts.
    # We do a loose check: cited source normalized text appears in any context normalized text.
    def normalize(s: str) -> str:
        return s.upper().replace("-", "").replace("_", "").replace(" ", "")

    factual_cases = [
        tc for tc in test_cases
        if tc["category"] in ("factual", "exact_clause", "cross_doc")
    ]
    citation_valid_count = 0
    citation_details = []
    for tc in factual_cases:
        ans = ans_by_id.get(tc["id"], {})
        cited = ans.get("cited_sources", [])
        contexts = ans.get("contexts", [])
        error = ans.get("error")

        if error or not cited:
            # Pipeline errored or returned no citations — mark as invalid
            citation_details.append({"id": tc["id"], "passed": False, "reason": "no_citations_or_error"})
            continue

        # Check each cited source has a matching context
        context_blob = normalize(" ".join(contexts))
        all_valid = True
        for source in cited:
            norm_source = normalize(source)
            # Partial match: at least 6 chars of the citation appear in contexts
            if len(norm_source) >= 6 and norm_source[:6] not in context_blob:
                all_valid = False
                break

        if all_valid:
            citation_valid_count += 1
        citation_details.append({
            "id": tc["id"],
            "passed": all_valid,
            "cited_sources": cited,
        })

    citation_validity = citation_valid_count / len(factual_cases) if factual_cases else 0.0

    # ── Error rate ────────────────────────────────────────────────────────────
    errored = [a for a in answers if a.get("error")]
    error_rate = len(errored) / len(answers) if answers else 0.0

    # ── Latency ───────────────────────────────────────────────────────────────
    latencies = [a["latency_ms"] for a in answers if not a.get("error") and not a.get("cache_hit")]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    return {
        "idk_accuracy": round(idk_accuracy, 4),
        "disclaimer_trigger_rate": round(disclaimer_rate, 4),
        "citation_validity": round(citation_validity, 4),
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "p95_latency_ms": round(p95_latency_ms, 1),
        "details": {
            "idk": idk_details,
            "disclaimer": disclaimer_details,
            "citation": citation_details,
            "errored_cases": [a["id"] for a in errored],
        },
    }


# ── RAGAS metrics ──────────────────────────────────────────────────────────────
def compute_ragas_metrics(test_cases: list, answers: list, ragas_llm, ragas_embeddings) -> dict:
    """
    Compute RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall.

    CRITICAL: Only run RAGAS on categories where it makes sense.
    - out_of_scope: skip faithfulness/context metrics — there ARE no relevant contexts
    - guardrail: skip — disclaimer content is not grounded in retrieved text by design
    - Run on: factual, exact_clause, cross_doc, ambiguous
    """
    # Filter to categories RAGAS should evaluate
    ragas_eligible = [
        tc for tc in test_cases
        if tc["category"] in ("factual", "exact_clause", "cross_doc", "ambiguous")
    ]

    ans_by_id = {a["id"]: a for a in answers}

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    skipped = []
    for tc in ragas_eligible:
        ans = ans_by_id.get(tc["id"], {})
        if ans.get("error"):
            skipped.append(tc["id"])
            continue
        contexts = ans.get("contexts", [])
        if not contexts:
            # No contexts retrieved — RAGAS context metrics will be 0, include anyway
            contexts = ["No context retrieved."]

        ragas_data["question"].append(tc["question"])
        ragas_data["answer"].append(ans.get("answer", ""))
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(tc["expected_answer"])

    if not ragas_data["question"]:
        print("[RAGAS] No eligible cases to evaluate. Check pipeline errors.")
        return {"error": "no_eligible_cases", "skipped": skipped}

    print(f"\n[RAGAS] Evaluating {len(ragas_data['question'])} cases "
          f"({len(skipped)} skipped due to pipeline errors)...")

    dataset = Dataset.from_dict(ragas_data)

    # Configure RAGAS to use our judge model
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = result.to_pandas()

    return {
        "faithfulness": round(float(scores["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"].mean()), 4),
        "context_precision": round(float(scores["context_precision"].mean()), 4),
        "context_recall": round(float(scores["context_recall"].mean()), 4),
        "n_evaluated": len(ragas_data["question"]),
        "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "per_case": scores[["faithfulness", "answer_relevancy",
                             "context_precision", "context_recall"]].to_dict(orient="records"),
    }


# ── DeepEval hallucination metric ──────────────────────────────────────────────
def compute_deepeval_metrics(test_cases: list, answers: list) -> dict:
    """
    Compute DeepEval hallucination metric.
    Only run on factual + exact_clause cases — guardrail and out_of_scope
    have different expected behaviors that don't map to hallucination detection.
    """
    eligible = [
        tc for tc in test_cases
        if tc["category"] in ("factual", "exact_clause", "cross_doc")
    ]
    ans_by_id = {a["id"]: a for a in answers}

    test_case_objects = []
    eligible_ids = []

    for tc in eligible:
        ans = ans_by_id.get(tc["id"], {})
        if ans.get("error") or not ans.get("answer"):
            continue

        contexts = ans.get("contexts", [])
        if not contexts:
            continue

        test_case_objects.append(LLMTestCase(
            input=tc["question"],
            actual_output=ans["answer"],
            expected_output=tc["expected_answer"],
            retrieval_context=contexts,
        ))
        eligible_ids.append(tc["id"])

    if not test_case_objects:
        return {"error": "no_eligible_cases"}

    print(f"\n[DeepEval] Evaluating hallucination on {len(test_case_objects)} cases...")

    hallucination_metric = HallucinationMetric(threshold=0.1)

    results = deepeval_evaluate(
        test_cases=test_case_objects,
        metrics=[hallucination_metric],
        print_results=False,
    )

    scores = []
    passed = 0
    for r in results.test_results:
        for m in r.metrics_data:
            if "hallucination" in m.name.lower():
                score = m.score if m.score is not None else 1.0
                scores.append(score)
                if m.success:
                    passed += 1

    hallucination_rate = sum(scores) / len(scores) if scores else 1.0

    return {
        "hallucination_rate": round(hallucination_rate, 4),
        "n_evaluated": len(test_case_objects),
        "n_passed": passed,
        "n_failed": len(test_case_objects) - passed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 4 Evaluation Runner")
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip pipeline execution, use saved raw answers")
    parser.add_argument("--category", default=None,
                        help="Run only one category (factual, cross_doc, out_of_scope, etc.)")
    parser.add_argument("--id", default=None,
                        help="Run a single test case by ID")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS metrics (faster, for debugging pipeline)")
    parser.add_argument("--skip-deepeval", action="store_true",
                        help="Skip DeepEval metrics")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    with open(dataset_path) as f:
        data = json.load(f)

    test_cases = data["test_cases"]

    # Apply filters
    if args.id:
        test_cases = [tc for tc in test_cases if tc["id"] == args.id]
        assert test_cases, f"No test case with id={args.id}"
    elif args.category:
        test_cases = [tc for tc in test_cases if tc["category"] == args.category]
        assert test_cases, f"No test cases with category={args.category}"

    print(f"\n{'='*60}")
    print(f"Phase 4 Evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_path} ({len(test_cases)} cases)")
    if args.dry_run:
        print("Mode: DRY RUN (using saved answers)")
    print(f"{'='*60}")

    # ── Output paths ──────────────────────────────────────────────────────────
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    saved_answers_path = str(results_dir / "raw_answers.json")
    output_path = results_dir / "latest.json"

    # ── Step 1: Collect pipeline answers ─────────────────────────────────────
    answers = collect_answers(test_cases, args.dry_run, saved_answers_path)

    # Save raw answers for --dry-run on next run
    if not args.dry_run:
        with open(saved_answers_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "n_cases": len(answers),
                "raw_answers": answers,
            }, f, indent=2)
        print(f"\n[saved] Raw answers → {saved_answers_path}")

    # ── Step 2: Custom metrics (no LLM judge needed) ──────────────────────────
    print("\n[custom metrics] Computing IDK accuracy, disclaimer rate, citation validity...")
    custom = compute_custom_metrics(test_cases, answers)

    print(f"  IDK accuracy:          {custom['idk_accuracy']:.2%}")
    print(f"  Disclaimer trigger:    {custom['disclaimer_trigger_rate']:.2%}")
    print(f"  Citation validity:     {custom['citation_validity']:.2%}")
    print(f"  Error rate:            {custom['error_rate']:.2%}")
    print(f"  Avg latency:           {custom['avg_latency_ms']:.0f}ms")
    print(f"  P95 latency:           {custom['p95_latency_ms']:.0f}ms")

    # ── Step 3: RAGAS metrics ─────────────────────────────────────────────────
    ragas_results = {}
    if not args.skip_ragas:
        ragas_llm, ragas_embeddings = _build_ragas_judge()
        ragas_results = compute_ragas_metrics(test_cases, answers, ragas_llm, ragas_embeddings)
        if "error" not in ragas_results:
            print(f"\n[RAGAS results]")
            print(f"  Faithfulness:          {ragas_results['faithfulness']:.4f}")
            print(f"  Answer relevancy:      {ragas_results['answer_relevancy']:.4f}")
            print(f"  Context precision:     {ragas_results['context_precision']:.4f}")
            print(f"  Context recall:        {ragas_results['context_recall']:.4f}")
    else:
        print("\n[RAGAS] Skipped (--skip-ragas)")

    # ── Step 4: DeepEval metrics ──────────────────────────────────────────────
    deepeval_results = {}
    if not args.skip_deepeval:
        deepeval_results = compute_deepeval_metrics(test_cases, answers)
        if "error" not in deepeval_results:
            print(f"\n[DeepEval results]")
            print(f"  Hallucination rate:    {deepeval_results['hallucination_rate']:.4f}")
            print(f"  Passed:                {deepeval_results['n_passed']}/{deepeval_results['n_evaluated']}")
    else:
        print("\n[DeepEval] Skipped (--skip-deepeval)")

    # ── Step 5: Assemble and save results ─────────────────────────────────────
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_version": data["metadata"]["version"],
        "n_cases": len(test_cases),
        "filters_applied": {
            "category": args.category,
            "id": args.id,
        },
        "metrics": {
            # Custom metrics
            "idk_accuracy": custom["idk_accuracy"],
            "disclaimer_trigger_rate": custom["disclaimer_trigger_rate"],
            "citation_validity": custom["citation_validity"],
            "error_rate": custom["error_rate"],
            "avg_latency_ms": custom["avg_latency_ms"],
            "p95_latency_ms": custom["p95_latency_ms"],
            # RAGAS
            "faithfulness": ragas_results.get("faithfulness"),
            "answer_relevancy": ragas_results.get("answer_relevancy"),
            "context_precision": ragas_results.get("context_precision"),
            "context_recall": ragas_results.get("context_recall"),
            # DeepEval
            "hallucination_rate": deepeval_results.get("hallucination_rate"),
        },
        "ragas_detail": ragas_results,
        "deepeval_detail": deepeval_results,
        "custom_detail": custom,
    }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[saved] Results → {output_path}")
    print(f"\n{'='*60}")
    print("Evaluation complete. Run check_gate.py to verify thresholds.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()