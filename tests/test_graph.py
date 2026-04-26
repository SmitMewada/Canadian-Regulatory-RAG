# tests/test_graph.py
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.graph import run_pipeline

if __name__ == "__main__":
    cases = [
        ("What does OSFI E-23 require for model validation?", None),
        ("What are the privacy obligations under PIPEDA?", "pipeda"),
        ("Should our bank use AI for credit decisions?", None),
        ("What are SEC requirements for algorithmic trading?", None),
    ]

    for query, doc_filter in cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        result = run_pipeline(query, doc_filter)
        answer = result["generated_answer"]

        if answer:
            print(f"Answer:     {answer.answer[:200]}...")
            print(f"Sources:    {answer.cited_sources}")
            print(f"Confidence: {answer.confidence}")
            print(f"Disclaimer: {answer.disclaimer is not None}")
            print(f"Review:     {answer.requires_human_review}")
        print(f"Eval score: {result['inline_eval_score']}")
        print(f"Citations:  {result['citation_valid']}")
        print(f"Retries:    {result['retry_count']}")