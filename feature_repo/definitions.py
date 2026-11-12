"""Feast definitions: departmental spending features for the budget-overrun model.

The same feature definitions feed offline training (point-in-time correct joins) and
online scoring (SQLite online store), so training and serving cannot drift apart.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "feast" / "dept_spend_features.parquet"

department = Entity(
    name="department", join_keys=["department"], value_type=ValueType.STRING, description="Government department"
)

dept_spend_source = FileSource(
    name="dept_spend_source",
    path=str(FEATURES_PATH),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

dept_spend_features = FeatureView(
    name="dept_spend_features",
    entities=[department],
    ttl=timedelta(days=62),
    schema=[
        Field(name="burn_rate_ratio", dtype=Float64, description="YTD spend / pro-rata budget"),
        Field(
            name="variance_to_budget_pct", dtype=Float64, description="(YTD spend - pro-rata budget) / annual budget"
        ),
        Field(name="open_commitments_ratio", dtype=Float64, description="Open commitments / remaining budget"),
        Field(name="spend_trend_3m", dtype=Float64, description="Last 3 months spend / previous 3 months"),
        Field(name="months_remaining", dtype=Int64),
        Field(name="prior_year_overrun", dtype=Int64),
    ],
    source=dept_spend_source,
    online=True,
)

overrun_model_v1 = FeatureService(name="overrun_model_v1", features=[dept_spend_features])
