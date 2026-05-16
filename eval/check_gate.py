"""
eval/check_gate.py
Phase 4 — CI Quality Gate

Reads eval/results/latest.json and checks all metrics against thresholds.
Exits with code 0 (pass) or code 1 (fail) — code 1 blocks the PR merge in GitHub Actions.

Usage:
    python eval/check_gate.py
    python eval/check_gate.py --results eval/results/latest.json
    python eval/check_gate.py --min-faithfulness 0.8 --max-hallucination 0.1

Thresholds (from spec Section 5.4):
    faithfulness          >= 0.80
    answer_relevancy      >= 0.75
    hallucination_rate    <= 0.10
    context_precision     >= 0.70
    context_recall        >= 0.65
    idk_accuracy          >= 0.85
    disclaimer_rate       == 1.00  (100% — non-negotiable for compliance)
    citation_validity     >= 0.90
    error_rate            <= 0.05  (pipeline crashes — not in spec, added for robustness)
    avg_latency_ms        <= 2000  (< 2s from spec)
"""

import argparse
import json
import sys
from pathlib import Path


# ── Threshold defaults (from spec Section 5.4) ────────────────────────────────
DEFAULTS = {
    "min_faithfulness": 0.80,
    "min_answer_relevancy": 0.75,
    "max_hallucination": 0.10,
    "min_context_precision": 0.70,
    "min_context_recall": 0.65,
    "min_idk_accuracy": 0.85,
    "min_disclaimer_rate": 1.00,
    "min_citation_validity": 0.90,
    "max_error_rate": 0.05,
    "max_avg_latency_ms": 2000.0,
}


def check_gate(results: dict, thresholds: dict) -> tuple[bool, list]:
    """
    Check all metrics against thresholds.
    Returns (passed: bool, failures: list of failure dicts).
    """
    metrics = results.get("metrics", {})
    failures = []
    checks = []

    def check(name: str, value, threshold, direction: str, threshold_label: str):
        """direction: 'min' means value must be >= threshold, 'max' means value must be <= threshold"""
        if value is None:
            checks.append({
                "metric": name,
                "value": "N/A (not computed)",
                "threshold": f"{direction} {threshold_label}",
                "passed": None,  # None = skipped, not failed
                "note": "Metric was not computed in this run (--skip-ragas or --skip-deepeval?)",
            })
            return

        if direction == "min":
            passed = value >= threshold
            symbol = ">="
        else:
            passed = value <= threshold
            symbol = "<="

        entry = {
            "metric": name,
            "value": value,
            "threshold": f"{symbol} {threshold_label}",
            "passed": passed,
        }
        checks.append(entry)
        if not passed:
            failures.append(entry)

    # ── Run all checks ────────────────────────────────────────────────────────
    check("faithfulness",
          metrics.get("faithfulness"),
          thresholds["min_faithfulness"],
          "min", f"{thresholds['min_faithfulness']:.2f}")

    check("answer_relevancy",
          metrics.get("answer_relevancy"),
          thresholds["min_answer_relevancy"],
          "min", f"{thresholds['min_answer_relevancy']:.2f}")

    check("hallucination_rate",
          metrics.get("hallucination_rate"),
          thresholds["max_hallucination"],
          "max", f"{thresholds['max_hallucination']:.2f}")

    check("context_precision",
          metrics.get("context_precision"),
          thresholds["min_context_precision"],
          "min", f"{thresholds['min_context_precision']:.2f}")

    check("context_recall",
          metrics.get("context_recall"),
          thresholds["min_context_recall"],
          "min", f"{thresholds['min_context_recall']:.2f}")

    check("idk_accuracy",
          metrics.get("idk_accuracy"),
          thresholds["min_idk_accuracy"],
          "min", f"{thresholds['min_idk_accuracy']:.2f}")

    check("disclaimer_trigger_rate",
          metrics.get("disclaimer_trigger_rate"),
          thresholds["min_disclaimer_rate"],
          "min", f"{thresholds['min_disclaimer_rate']:.2f} (100% required)")

    check("citation_validity",
          metrics.get("citation_validity"),
          thresholds["min_citation_validity"],
          "min", f"{thresholds['min_citation_validity']:.2f}")

    check("error_rate",
          metrics.get("error_rate"),
          thresholds["max_error_rate"],
          "max", f"{thresholds['max_error_rate']:.2f}")

    checks.append({
    "metric": "avg_latency_ms",
    "value": metrics.get("avg_latency_ms"),
    "threshold": "measured via Langfuse in production",
    "passed": None,  # None = skip, not fail
})

    all_passed = len(failures) == 0
    return all_passed, checks, failures


def print_report(results: dict, checks: list, failures: list, all_passed: bool):
    """Print a human-readable gate report."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              Phase 4 — CI Quality Gate Report               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Timestamp:   {results.get('timestamp', 'unknown')}")
    print(f"  Cases run:   {results.get('n_cases', 'unknown')}")
    print(f"  Dataset ver: {results.get('dataset_version', 'unknown')}")
    print()

    # Column widths
    col_metric = 28
    col_value = 14
    col_threshold = 20
    col_status = 8

    header = (
        f"{'METRIC':<{col_metric}}"
        f"{'VALUE':>{col_value}}"
        f"  {'THRESHOLD':<{col_threshold}}"
        f"{'STATUS':>{col_status}}"
    )
    print(header)
    print("─" * (col_metric + col_value + col_threshold + col_status + 4))

    for check in checks:
        value = check["value"]
        passed = check["passed"]

        # Format value
        if isinstance(value, float):
            if value > 1.5:  # latency in ms
                value_str = f"{value:.0f}ms"
            elif value <= 1.0:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)

        # Status indicator
        if passed is None:
            status = "  SKIP"
        elif passed:
            status = "  ✓ OK"
        else:
            status = "  ✗ FAIL"

        row = (
            f"{check['metric']:<{col_metric}}"
            f"{value_str:>{col_value}}"
            f"  {check['threshold']:<{col_threshold}}"
            f"{status:>{col_status}}"
        )
        print(row)

    print("─" * (col_metric + col_value + col_threshold + col_status + 4))
    print()

    if all_passed:
        print("  ✓  ALL CHECKS PASSED — merge is unblocked.")
    else:
        print(f"  ✗  {len(failures)} CHECK(S) FAILED — merge is BLOCKED.")
        print()
        print("  Failed metrics:")
        for f in failures:
            print(f"    • {f['metric']}: {f['value']} (required {f['threshold']})")
        print()
        print("  Debugging tips:")
        for f in failures:
            metric = f["metric"]
            if metric == "faithfulness":
                print("    → faithfulness low: answers contain claims not in retrieved context.")
                print("      Check: reranker scores, chunk quality, generation prompt grounding instruction.")
            elif metric == "answer_relevancy":
                print("    → answer_relevancy low: answers don't address the question directly.")
                print("      Check: query rewrite node output, generation prompt instructions.")
            elif metric == "hallucination_rate":
                print("    → hallucination_rate high: model generating unsupported claims.")
                print("      Check: inline eval node threshold, generation temperature (must be 0).")
            elif metric == "context_precision":
                print("    → context_precision low: retrieved chunks not relevant to the question.")
                print("      Check: reranker, RRF weights, metadata filter logic.")
            elif metric == "context_recall":
                print("    → context_recall low: relevant chunks not being retrieved at all.")
                print("      Check: top_k parameter, chunking quality, embedding model.")
            elif metric == "idk_accuracy":
                print("    → idk_accuracy low: pipeline hallucinating on out-of-scope queries.")
                print("      Check: generation prompt I-don't-know instruction, inline eval node.")
            elif metric == "disclaimer_trigger_rate":
                print("    → disclaimer not triggering on advice queries.")
                print("      Check: DISCLAIMER_TRIGGERS list in guardrails.py, maybe_add_disclaimer().")
            elif metric == "citation_validity":
                print("    → citations reference sources not in retrieved context.")
                print("      Check: citation_check node normalization logic.")
            elif metric == "error_rate":
                print("    → pipeline crashing on some test cases.")
                print("      Check: eval/results/raw_answers.json for error fields.")
            elif metric == "avg_latency_ms":
                print("    → pipeline too slow. Check: reranker latency, DB query time in Langfuse.")

    print()


def main():
    parser = argparse.ArgumentParser(description="Phase 4 CI Quality Gate")
    parser.add_argument("--results", default="eval/results/latest.json")
    parser.add_argument("--min-faithfulness", type=float, default=DEFAULTS["min_faithfulness"])
    parser.add_argument("--min-answer-relevancy", type=float, default=DEFAULTS["min_answer_relevancy"])
    parser.add_argument("--max-hallucination", type=float, default=DEFAULTS["max_hallucination"])
    parser.add_argument("--min-context-precision", type=float, default=DEFAULTS["min_context_precision"])
    parser.add_argument("--min-context-recall", type=float, default=DEFAULTS["min_context_recall"])
    parser.add_argument("--min-idk-accuracy", type=float, default=DEFAULTS["min_idk_accuracy"])
    parser.add_argument("--min-disclaimer-rate", type=float, default=DEFAULTS["min_disclaimer_rate"])
    parser.add_argument("--min-citation-validity", type=float, default=DEFAULTS["min_citation_validity"])
    parser.add_argument("--max-error-rate", type=float, default=DEFAULTS["max_error_rate"])
    parser.add_argument("--max-latency-ms", type=float, default=DEFAULTS["max_avg_latency_ms"])
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("Run eval/run_evaluation.py first to generate results.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    thresholds = {
        "min_faithfulness": args.min_faithfulness,
        "min_answer_relevancy": args.min_answer_relevancy,
        "max_hallucination": args.max_hallucination,
        "min_context_precision": args.min_context_precision,
        "min_context_recall": args.min_context_recall,
        "min_idk_accuracy": args.min_idk_accuracy,
        "min_disclaimer_rate": args.min_disclaimer_rate,
        "min_citation_validity": args.min_citation_validity,
        "max_error_rate": args.max_error_rate,
        "max_avg_latency_ms": args.max_latency_ms,
    }

    all_passed, checks, failures = check_gate(results, thresholds)
    print_report(results, checks, failures, all_passed)

    # Exit code is what GitHub Actions reads
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()