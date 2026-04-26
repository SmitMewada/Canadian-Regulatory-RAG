from dotenv import load_dotenv
load_dotenv()

from src.pipeline.nodes.retrieve import retrieve_node
from src.pipeline.nodes.rerank import rerank_node
from src.pipeline.nodes.generate import generate_node
from src.pipeline.nodes.inline_eval import inline_eval_node
from src.pipeline.nodes.citation_check import citation_check_node
from src.models.schemas import GraphState
from src.models.schemas import RegulatoryAnswer


def make_state(query: str, doc_filter: str = None) -> GraphState:
    return GraphState(
        original_query=query,
        rewritten_query=query,
        document_filter=doc_filter,
        retrieved_chunks=[],
        reranked_chunks=[],
        generated_answer=None,
        inline_eval_score=None,
        citation_valid=None,
        retry_count=0,
        langfuse_trace_id=None,
        cache_hit=False,
    )

if __name__ == "__main__":
    cases = [
    # Original cases
    ("What does OSFI E-23 require for model validation?", None),
    ("What are the privacy obligations under PIPEDA?", "pipeda"),
    ("What are SEC requirements for algorithmic trading?", None),
    
    # Edge cases
    ("What are the specific penalties under PIPEDA for non-compliance?", "pipeda"),  # answer likely not in chunks
    ("What does E-23 say about quantum computing risk models?", None),  # very specific, probably not in corpus
    ("What are all 10 principles of PIPEDA?", "pipeda"),  # asks for complete list — chunks probably partial
]

    for query, doc_filter in cases:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        state = make_state(query, doc_filter)
        state = retrieve_node(state)
        state = rerank_node(state)
        state = generate_node(state)
        state = inline_eval_node(state)
        state = citation_check_node(state)

        answer = state["generated_answer"]
        print(f"Eval score:     {state['inline_eval_score']}/5")
        print(f"Citation valid: {state['citation_valid']}")
        print(f"Confidence:     {answer.confidence}")
        print(f"Needs review:   {answer.requires_human_review}")
        
     # ── Hallucination injection test ──────────────────────────────

    print(f"\n{'='*60}")
    print("HALLUCINATION INJECTION TEST")

    state = make_state("What does OSFI E-23 require for model validation?", None)
    state = retrieve_node(state)
    state = rerank_node(state)

    # Inject a completely fabricated answer
    state["generated_answer"] = RegulatoryAnswer(
        answer="OSFI E-23 requires all banks to use quantum encryption for model validation and mandates a minimum of 500 human reviewers per model deployment. All models must be validated on Mars by 2025.",
        cited_sources=["OSFI E23"],
        confidence="high",
        requires_human_review=False,
    )

    state = inline_eval_node(state)
    print(f"Hallucination test score: {state['inline_eval_score']}/5")
    print(f"Expected: 1 or 2")
    if state['inline_eval_score'] >= 4:
        print("⚠️  SELF-CONSISTENCY BIAS CONFIRMED — eval failed to catch hallucination")
    else:
        print("✓ Eval correctly flagged hallucination")