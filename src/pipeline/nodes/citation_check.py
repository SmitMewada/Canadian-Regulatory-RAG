from src.models.schemas import GraphState

def citation_check_node(state: GraphState) -> GraphState:
    """
    Verifies that every cited source maps to an actual retrieved chunk.
    Simple rule: cited_source must contain a document_id substring.
    If any citation is invalid, downgrade confidence.
    """
    answer = state["generated_answer"]
    chunks = state["reranked_chunks"] or state["retrieved_chunks"]

    if not answer or not answer.cited_sources:
        print("  [citation_check] no citations to verify")
        return {**state, "citation_valid": True}

    # Get all document_ids from retrieved chunks
    retrieved_doc_ids = {c.document_id.upper().replace("_", " ")
                        for c in chunks}

    invalid = []
    for citation in answer.cited_sources:
        citation_upper = citation.upper()
        # Check if any retrieved doc_id is a substring of the citation
        matched = any(doc_id in citation_upper
                     for doc_id in retrieved_doc_ids)
        if not matched:
            invalid.append(citation)

    if invalid:
        print(f"  [citation_check] invalid citations: {invalid}")
        answer.confidence = "low"
        answer.requires_human_review = True
        return {**state, "citation_valid": False, "generated_answer": answer}

    print(f"  [citation_check] all {len(answer.cited_sources)} citations valid ✓")
    return {**state, "citation_valid": True}