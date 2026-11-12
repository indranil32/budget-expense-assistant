"""Score departments in real time from the Feast online store with the registered model.

python -m budget_assistant.model.serve --as-of 2025-06-30 revenue treasury it
"""

from __future__ import annotations

import argparse
import datetime as dt

import mlflow
import pandas as pd

from ..features.build import FEATURE_COLUMNS, feature_store
from .train import MODEL_NAME, tracking_uri


def load_model(version: str = "latest"):  # type: ignore[no-untyped-def]
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")


def score(departments: list[str], as_of: dt.datetime, model_version: str = "latest") -> pd.DataFrame:
    store = feature_store()
    # Push the latest feature values up to `as_of` into the online store (SQLite here; Redis/Cosmos in production).
    store.materialize(start_date=as_of - dt.timedelta(days=62), end_date=as_of)
    online = store.get_online_features(
        features=store.get_feature_service("overrun_model_v1"),
        entity_rows=[{"department": d} for d in departments],
    ).to_df()
    model = load_model(model_version)
    probabilities = model.predict(online[FEATURE_COLUMNS])  # (n, 2) from predict_proba
    online["overrun_probability"] = pd.DataFrame(probabilities).iloc[:, 1].round(3).to_numpy()
    return online[["department", "overrun_probability", *FEATURE_COLUMNS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("departments", nargs="+")
    parser.add_argument("--as-of", default="2025-06-30")
    args = parser.parse_args()
    as_of = dt.datetime.fromisoformat(args.as_of).replace(tzinfo=dt.UTC)
    print(score(args.departments, as_of).to_string(index=False))


if __name__ == "__main__":
    main()
