from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from budget_assistant.classify import BaselineClassifier
from budget_assistant.experiments.bayesian_ab import assign_variant
from budget_assistant.llm import HashingEmbedder
from budget_assistant.paths import CLAIMS_CSV
from budget_assistant.rag.chunking import by_heading, load_documents
from budget_assistant.rag.retrieval import LocalHybridRetriever

TODAY = dt.date(2026, 11, 10)


@pytest.fixture(scope="session")
def retriever() -> LocalHybridRetriever:
    return LocalHybridRetriever(by_heading(load_documents()), HashingEmbedder(), "hybrid")


@pytest.fixture(scope="session")
def classifier() -> BaselineClassifier:
    claims = pd.read_csv(CLAIMS_CSV)
    return BaselineClassifier(claims[claims["split"] == "train"])


def user_in_variant(variant: str) -> str:
    return next(f"user-{i}" for i in range(1000) if assign_variant(f"user-{i}") == variant)
