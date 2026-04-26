# tests/test_cache.py
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.graph import run_pipeline

if __name__ == "__main__":
    query = "What does OSFI E-23 require for model validation?"

    print("Run 1 — expect MISS, full pipeline")
    result1 = run_pipeline(query)
    print(f"cache_hit={result1['cache_hit']}")
    print(f"answer={result1['generated_answer'].answer[:100]}...")

    print(f"\n{'='*60}")
    print("Run 2 — expect HIT, cached response")
    result2 = run_pipeline(query)
    print(f"cache_hit={result2['cache_hit']}")
    print(f"answer={result2['generated_answer'].answer[:100]}...")

    print(f"\n{'='*60}")
    print("Run 3 — semantically similar query, expect HIT")
    similar_query = "What are OSFI E-23 model validation requirements?"
    result3 = run_pipeline(similar_query)
    print(f"cache_hit={result3['cache_hit']}")