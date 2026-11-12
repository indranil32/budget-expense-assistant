"""Expense-claim classification into spending categories.

Three interchangeable classifiers, compared by ``finetune.benchmark``:
  BaselineClassifier   TF-IDF + logistic regression (cheap, no LLM) — the bar an LLM must beat
  FewShotClassifier    general chat model + category definitions + examples in the prompt
  FineTunedClassifier  a model fine-tuned on labelled claims (short prompt, no examples)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel, Field

from .llm import Chat

Category = Literal["travel", "training", "it_equipment", "office_supplies", "hospitality", "professional_services"]
CATEGORIES: tuple[str, ...] = (
    "travel",
    "training",
    "it_equipment",
    "office_supplies",
    "hospitality",
    "professional_services",
)


class ClaimCategory(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)


@dataclass
class Prediction:
    category: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class Classifier(Protocol):
    name: str

    def predict(self, description: str) -> Prediction: ...


DEFINITIONS = """Categories:
- travel: flights, hotels, taxis, visas, per-diem meals while travelling
- training: courses, exams, certifications, workshops (including their fees)
- it_equipment: hardware and accessories (docks, monitors, chargers, headsets)
- office_supplies: paper, toner, stationery, folders
- hospitality: meals, coffee or catering hosted for external parties or team events
- professional_services: legal, translation, consultancy, notary fees"""

FINE_TUNED_SYSTEM = "Classify the expense claim into one spending category. Reply with JSON."


def few_shot_system(examples: Sequence[tuple[str, str]]) -> str:
    shots = "\n".join(f'- "{text}" -> {label}' for text, label in examples)
    return (
        f"Classify the expense claim into exactly one category.\n{DEFINITIONS}\n\nExamples:\n{shots}\nReply with JSON."
    )


class LLMClassifier:
    def __init__(self, chat: Chat, system: str, name: str) -> None:
        self.chat, self.system, self.name = chat, system, name

    def predict(self, description: str) -> Prediction:
        result = self.chat.complete(self.system, description, schema=ClaimCategory)
        try:
            category = result.parse(ClaimCategory).category
        except ValueError:
            category = "unparseable"
        return Prediction(category, result.latency_ms, result.input_tokens, result.output_tokens)


def few_shot_classifier(chat: Chat, train: pd.DataFrame, shots_per_class: int = 2) -> LLMClassifier:
    examples = [
        (row.description, row.category)
        for _, grp in train.groupby("category")
        for row in grp.head(shots_per_class).itertuples()
    ]
    return LLMClassifier(chat, few_shot_system(examples), "few_shot")


def fine_tuned_classifier(chat: Chat) -> LLMClassifier:
    return LLMClassifier(chat, FINE_TUNED_SYSTEM, "fine_tuned")


class BaselineClassifier:
    name = "tfidf_logreg"

    def __init__(self, train: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=5.0)
        )
        self.model.fit(train["description"], train["category"])

    def predict(self, description: str) -> Prediction:
        start = time.perf_counter()
        label = str(self.model.predict([description])[0])
        return Prediction(label, (time.perf_counter() - start) * 1000)
