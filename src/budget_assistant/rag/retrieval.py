"""Hybrid retrieval: keyword (BM25) + vector, fused with Reciprocal Rank Fusion.

- ``LocalHybridRetriever``: in-memory BM25 + vector search (development, tests, CI).
- ``AzureSearchRetriever``: Azure AI Search hybrid query (keyword + HNSW vector in one
  request; the service fuses them with RRF), for production.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from ..llm import Embedder
from .chunking import Chunk

Mode = Literal["keyword", "vector", "hybrid"]
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class LocalHybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, mode: Mode = "hybrid") -> None:
        self.chunks = list(chunks)
        self.mode = mode
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._vectors = embedder.embed([c.text for c in self.chunks])

    def _keyword_rank(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)]

    def _vector_rank(self, query: str) -> list[int]:
        sims = self._vectors @ self.embedder.embed([query])[0]
        return [int(i) for i in np.argsort(-sims)]

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        if self.mode == "keyword":
            order = self._keyword_rank(query)
        elif self.mode == "vector":
            order = self._vector_rank(query)
        else:
            order = reciprocal_rank_fusion([self._keyword_rank(query), self._vector_rank(query)])
        return [self.chunks[i] for i in order[:k]]


class AzureSearchRetriever:
    """Hybrid search against the index built by ``index_azure``."""

    def __init__(
        self, embedder: Embedder, endpoint: str | None = None, index: str | None = None, client: Any = None
    ) -> None:
        import os

        self.embedder = embedder
        if client is None:
            from azure.identity import DefaultAzureCredential
            from azure.search.documents import SearchClient

            client = SearchClient(
                endpoint or os.environ["AZURE_SEARCH_ENDPOINT"],
                index or os.environ.get("AZURE_SEARCH_INDEX", "policies"),
                DefaultAzureCredential(),
            )
        self.client = client

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        from azure.search.documents.models import VectorizedQuery

        vector = self.embedder.embed([query])[0].tolist()
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k * 5, fields="embedding")],
            select=["id", "doc", "section", "content"],
            top=k,
        )
        return [Chunk(r["id"], r["doc"], r["section"], r["content"]) for r in results]
