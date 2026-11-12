"""Build monthly departmental spending features and register them in Feast.

    python -m budget_assistant.features.build

Each row is a department at a month-end snapshot. Features only use data up to that
month, and the label (year-end overrun) is stored separately, so nothing leaks from
the future into training.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..paths import FEAST_DATA, FEATURE_REPO, SPEND_CSV

FEATURE_COLUMNS = [
    "burn_rate_ratio",
    "variance_to_budget_pct",
    "open_commitments_ratio",
    "spend_trend_3m",
    "months_remaining",
    "prior_year_overrun",
]
FEATURE_REFS = [f"dept_spend_features:{c}" for c in FEATURE_COLUMNS]


def compute_features(spend: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, labels), both keyed by department + month-end event_timestamp."""
    df = spend.sort_values(["department", "year", "month"]).copy()
    grp = df.groupby(["department", "year"])
    df["ytd_spend"] = grp["spend_aed"].cumsum()
    df["pro_rata_budget"] = df["annual_budget_aed"] * df["month"] / 12
    df["burn_rate_ratio"] = df["ytd_spend"] / df["pro_rata_budget"]
    df["variance_to_budget_pct"] = (df["ytd_spend"] - df["pro_rata_budget"]) / df["annual_budget_aed"]
    remaining = (df["annual_budget_aed"] - df["ytd_spend"]).clip(lower=df["annual_budget_aed"] * 0.01)
    df["open_commitments_ratio"] = df["open_commitments_aed"] / remaining
    last3 = grp["spend_aed"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    prev3 = grp["spend_aed"].transform(lambda s: s.shift(3).rolling(3, min_periods=1).mean())
    df["spend_trend_3m"] = (last3 / prev3).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["months_remaining"] = 12 - df["month"]

    year_total = df.groupby(["department", "year"])["spend_aed"].sum().rename("year_total").reset_index()
    budgets = df.groupby(["department", "year"])["annual_budget_aed"].first().reset_index()
    outcome = year_total.merge(budgets, on=["department", "year"])
    outcome["overran"] = (outcome["year_total"] > outcome["annual_budget_aed"]).astype(int)
    prior = outcome[["department", "year", "overran"]].assign(year=lambda o: o["year"] + 1)
    df = df.merge(prior.rename(columns={"overran": "prior_year_overrun"}), on=["department", "year"], how="left")
    df["prior_year_overrun"] = df["prior_year_overrun"].fillna(0).astype(int)
    df = df.merge(outcome[["department", "year", "overran"]], on=["department", "year"])

    df["event_timestamp"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}, utc=True
    ) + pd.offsets.MonthEnd(0)
    df["created_timestamp"] = pd.Timestamp(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))

    features = df[["department", "event_timestamp", "created_timestamp", *FEATURE_COLUMNS]].reset_index(drop=True)
    labels = df[["department", "event_timestamp", "year", "month", "overran"]].rename(
        columns={"overran": "will_overrun"}
    )
    return features, labels.reset_index(drop=True)


def feature_store():  # type: ignore[no-untyped-def]
    from feast import FeatureStore

    return FeatureStore(repo_path=str(FEATURE_REPO))


def build_and_register() -> tuple[pd.DataFrame, pd.DataFrame]:
    import importlib.util

    FEAST_DATA.mkdir(parents=True, exist_ok=True)
    features, labels = compute_features(pd.read_csv(SPEND_CSV))
    features.to_parquet(FEAST_DATA / "dept_spend_features.parquet", index=False)
    labels.to_parquet(FEAST_DATA / "labels.parquet", index=False)

    spec = importlib.util.spec_from_file_location("feast_definitions", FEATURE_REPO / "definitions.py")
    assert spec and spec.loader
    defs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defs)
    store = feature_store()
    store.apply([defs.department, defs.dept_spend_source, defs.dept_spend_features, defs.overrun_model_v1])
    return features, labels


if __name__ == "__main__":
    f, lab = build_and_register()
    print(f"{len(f)} feature rows, {lab['will_overrun'].mean():.0%} of department-years overran; registered in Feast")
