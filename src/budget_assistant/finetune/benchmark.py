"""Benchmark claim classifiers on the held-out test set: accuracy, macro-F1, latency, cost.

    python -m budget_assistant.finetune.benchmark          # offline: TF-IDF baseline only
    python -m budget_assistant.finetune.benchmark --live   # + few-shot and fine-tuned Azure deployments

The question it answers: is fine-tuning worth it versus prompting a general model,
and does either beat a cheap classical baseline?
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..classify import BaselineClassifier, Classifier, few_shot_classifier, fine_tuned_classifier
from ..paths import RESULTS
from .prepare import OUT, write


def run(classifier: Classifier, test: pd.DataFrame, price_in: float, price_out: float) -> dict[str, float]:
    preds = [classifier.predict(d) for d in test["description"]]
    labels = [p.category for p in preds]
    cost = sum(p.input_tokens * price_in + p.output_tokens * price_out for p in preds) / 1_000_000
    return {
        "accuracy": round(accuracy_score(test["category"], labels), 3),
        "macro_f1": round(f1_score(test["category"], labels, average="macro"), 3),
        "mean_latency_ms": round(statistics.mean(p.latency_ms for p in preds), 1),
        "cost_per_1k_claims_usd": round(cost / len(preds) * 1000, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    write()
    train, test = pd.read_csv(OUT / "train.csv"), pd.read_csv(OUT / "test.csv")
    price_in = float(os.environ.get("PRICE_INPUT_PER_M", "0"))
    price_out = float(os.environ.get("PRICE_OUTPUT_PER_M", "0"))

    classifiers: list[Classifier] = [BaselineClassifier(train)]
    if args.live:
        from ..llm import AzureChat

        classifiers.append(few_shot_classifier(AzureChat(), train))
        classifiers.append(fine_tuned_classifier(AzureChat(os.environ["AZURE_OPENAI_FINETUNED_DEPLOYMENT"])))

    table = {c.name: run(c, test, price_in, price_out) for c in classifiers}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ("classifier_live.json" if args.live else "classifier_offline.json")).write_text(
        json.dumps(table, indent=2)
    )
    print(json.dumps({"test_size": len(test), **table}, indent=2))


if __name__ == "__main__":
    main()
