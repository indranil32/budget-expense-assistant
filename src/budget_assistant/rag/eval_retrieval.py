"""Retrieval evaluation: hit rate@k and MRR for each chunking strategy and search mode.

    python -m budget_assistant.rag.eval_retrieval            # offline embedder (CI)
    python -m budget_assistant.rag.eval_retrieval --azure    # Azure OpenAI embeddings

A hit means a retrieved chunk comes from the expected document and section (for
fixed-size chunks, which have no section, the expected section text must appear in it).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ..llm import AzureEmbedder, Embedder, HashingEmbedder
from ..paths import RESULTS, ROOT
from .chunking import STRATEGIES, Chunk, load_documents
from .retrieval import LocalHybridRetriever

GOLDEN = ROOT / "evals" / "retrieval_golden.jsonl"
MODES = ("keyword", "vector", "hybrid")


def is_hit(chunk: Chunk, case: dict[str, str]) -> bool:
    if chunk.doc != case["doc"]:
        return False
    return chunk.section == case["section"] if chunk.section else case["section"].lower() in chunk.text.lower()


def evaluate(embedder: Embedder, k: int = 3) -> dict[str, dict[str, float]]:
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    docs = load_documents()
    table: dict[str, dict[str, float]] = {}
    for strategy, chunker in STRATEGIES.items():
        chunks = chunker(docs)
        for mode in MODES:
            retriever = LocalHybridRetriever(chunks, embedder, mode)  # type: ignore[arg-type]
            hits1 = hitsk = 0
            rr = 0.0
            for case in cases:
                results = retriever.search(case["q"], k=k)
                ranks = [i for i, c in enumerate(results) if is_hit(c, case)]
                hits1 += bool(ranks and ranks[0] == 0)
                hitsk += bool(ranks)
                rr += 1 / (ranks[0] + 1) if ranks else 0.0
            n = len(cases)
            table[f"{strategy}/{mode}"] = {
                "hit@1": round(hits1 / n, 3),
                f"hit@{k}": round(hitsk / n, 3),
                "mrr": round(rr / n, 3),
                "chunks": len(chunks),
            }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--azure", action="store_true")
    args = parser.parse_args()
    embedder: Any = AzureEmbedder() if args.azure else HashingEmbedder()
    table = evaluate(embedder)
    RESULTS.mkdir(exist_ok=True)
    name = "retrieval_azure.json" if args.azure else "retrieval_offline.json"
    (RESULTS / name).write_text(json.dumps(table, indent=2))
    print(f"{'strategy/mode':28} hit@1  hit@3  mrr")
    for key, row in table.items():
        print(f"{key:28} {row['hit@1']:.2f}   {row['hit@3']:.2f}   {row['mrr']:.2f}")


if __name__ == "__main__":
    main()
