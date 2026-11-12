import os

import pandas as pd
import pytest

from budget_assistant.features.build import FEATURE_COLUMNS, compute_features
from budget_assistant.paths import SPEND_CSV


def test_features_use_only_past_data() -> None:
    spend = pd.read_csv(SPEND_CSV)
    base, _ = compute_features(spend)
    changed = spend.copy()
    future = (changed["department"] == "revenue") & (changed["year"] == 2024) & (changed["month"] == 12)
    changed.loc[future, "spend_aed"] *= 10  # a huge December change...
    after, _ = compute_features(changed)
    mask = (base["department"] == "revenue") & (base["event_timestamp"] < "2024-12-01")
    # ...must not change any earlier snapshot of the same year
    pd.testing.assert_frame_equal(base.loc[mask, FEATURE_COLUMNS], after.loc[mask, FEATURE_COLUMNS])


def test_labels_match_year_end_outcome() -> None:
    spend = pd.read_csv(SPEND_CSV)
    _, labels = compute_features(spend)
    row = spend[(spend["department"] == "audit") & (spend["year"] == 2023)]
    expected = int(row["spend_aed"].sum() > row["annual_budget_aed"].iloc[0])
    assert set(labels[(labels["department"] == "audit") & (labels["year"] == 2023)]["will_overrun"]) == {expected}


@pytest.mark.slow
def test_train_registers_model_and_writes_monitoring_windows(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    from budget_assistant.model.train import train
    from budget_assistant.paths import MONITORING

    result = train()
    assert result["valid_roc_auc"] > 0.75
    assert (MONITORING / "reference.parquet").exists() and (MONITORING / "current.parquet").exists()
    assert os.path.exists(f"{tmp_path}/mlflow.db")
