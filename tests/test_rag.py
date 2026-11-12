from budget_assistant.llm import HashingEmbedder
from budget_assistant.rag.chunking import by_heading, fixed_size, load_documents
from budget_assistant.rag.eval_retrieval import evaluate
from budget_assistant.rag.retrieval import LocalHybridRetriever, reciprocal_rank_fusion


def test_heading_chunks_are_self_describing() -> None:
    chunks = by_heading(load_documents())
    per_diem = next(c for c in chunks if c.section == "Per diem")
    assert per_diem.doc == "travel-policy"
    assert per_diem.text.startswith("Official Travel Policy (Circular FIN-07) - Per diem:")
    assert all(c.text for c in fixed_size(load_documents()))


def test_rrf_rewards_agreement_between_rankers() -> None:
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1])
    assert reciprocal_rank_fusion([[5, 1], [1, 5], [1, 7]])[0] == 1


def test_hybrid_retrieves_the_right_section(retriever: LocalHybridRetriever) -> None:
    top = retriever.search("Who approves a claim of AED 12,000?", k=3)
    assert ("expense-claims", "Approval levels") in [(c.doc, c.section) for c in top]


def test_retrieval_quality_gate() -> None:
    table = evaluate(HashingEmbedder())
    assert table["by_heading/hybrid"]["hit@3"] >= 0.85
    assert table["by_heading/hybrid"]["mrr"] >= table["fixed_size/keyword"]["mrr"]
