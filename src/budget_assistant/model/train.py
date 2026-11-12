"""Train the budget-overrun model from Feast features and track it in MLflow.

    python -m budget_assistant.model.train

Question the model answers: "Given a department's spending so far this year, will it
exceed its annual budget by year end?"  Finance can then act before Q3 variance reports.

- Features come from Feast with point-in-time joins (no future leakage).
- Time-based split: train on FY2023, validate on FY2024. FY2025 is held back as
  "production" data for drift monitoring.
- Params, metrics, the feature list and the model are logged to MLflow, and the model
  is registered as ``budget-overrun``.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ..features.build import FEATURE_COLUMNS, build_and_register, feature_store
from ..paths import FEAST_DATA, MONITORING, ROOT

MODEL_NAME = "budget-overrun"
PARAMS = {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05, "random_state": 7}


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def training_frame() -> pd.DataFrame:
    """Labels joined to Feast features as of each label's timestamp."""
    store = feature_store()
    labels = pd.read_parquet(FEAST_DATA / "labels.parquet")
    job = store.get_historical_features(entity_df=labels, features=store.get_feature_service("overrun_model_v1"))
    frame = job.to_df()
    return frame.dropna(subset=FEATURE_COLUMNS).sort_values(["department", "event_timestamp"]).reset_index(drop=True)


def train(register: bool = True) -> dict[str, Any]:
    build_and_register()
    frame = training_frame()
    train_df = frame[frame["year"] == 2023]
    valid_df = frame[frame["year"] == 2024]
    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["will_overrun"]
    x_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df["will_overrun"]

    model = GradientBoostingClassifier(**PARAMS).fit(x_train, y_train)
    proba = model.predict_proba(x_valid)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "valid_roc_auc": roc_auc_score(y_valid, proba),
        "valid_precision": precision_score(y_valid, pred),
        "valid_recall": recall_score(y_valid, pred),
        "valid_f1": f1_score(y_valid, pred),
        # Early warning matters most: how good is the model by mid-year (months 4-6)?
        "valid_roc_auc_h1": roc_auc_score(
            valid_df.loc[valid_df["month"].between(4, 6), "will_overrun"],
            model.predict_proba(valid_df.loc[valid_df["month"].between(4, 6), FEATURE_COLUMNS])[:, 1],
        ),
    }

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("budget-overrun")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {**PARAMS, "train_years": "2023", "valid_years": "2024", "feature_service": "overrun_model_v1"}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"features": FEATURE_COLUMNS}, "features.json")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(x_train, model.predict_proba(x_train)),
            pyfunc_predict_fn="predict_proba",  # serve probabilities, not 0/1 labels
            input_example=x_train.head(3),
            registered_model_name=MODEL_NAME if register else None,
            # skops serialisation: we trained this model ourselves, so trust only the tree node type it needs.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    # Reference window for drift monitoring = the data the model learned from.
    MONITORING.mkdir(parents=True, exist_ok=True)
    frame[frame["year"].isin([2023, 2024])][FEATURE_COLUMNS].to_parquet(MONITORING / "reference.parquet")
    frame[frame["year"] == 2025][FEATURE_COLUMNS].to_parquet(MONITORING / "current.parquet")
    return {"run_id": run_id, **{k: round(v, 3) for k, v in metrics.items()}}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
