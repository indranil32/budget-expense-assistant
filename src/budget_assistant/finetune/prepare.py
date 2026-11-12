"""Split labelled claims and write fine-tuning files in chat format.

    python -m budget_assistant.finetune.prepare

Outputs data/finetune/{train,valid}.jsonl (chat-completions format) and test.csv
(held out, never seen by fine-tuning or few-shot examples).
"""

from __future__ import annotations

import json

import pandas as pd

from ..classify import FINE_TUNED_SYSTEM
from ..paths import CLAIMS_CSV, DATA

OUT = DATA / "finetune"


def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the fixed split from the data generator: the test set includes wording never seen in training."""
    claims = pd.read_csv(CLAIMS_CSV)
    return tuple(claims[claims["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))  # type: ignore[return-value]


def to_chat_record(description: str, category: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": FINE_TUNED_SYSTEM},
            {"role": "user", "content": description},
            {"role": "assistant", "content": json.dumps({"category": category, "confidence": 1.0})},
        ]
    }


def write() -> dict[str, int]:
    train, valid, test = split()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("valid", valid)):
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for row in frame.itertuples():
                fh.write(json.dumps(to_chat_record(row.description, row.category)) + "\n")
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    return {"train": len(train), "valid": len(valid), "test": len(test)}


if __name__ == "__main__":
    print(write())
