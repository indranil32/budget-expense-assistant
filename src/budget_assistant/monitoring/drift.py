"""Data-drift monitoring with Evidently.

    python -m budget_assistant.monitoring.drift                 # FY2025 vs training window
    python -m budget_assistant.monitoring.drift --fail-on-drift # exit 1 when drift is detected (scheduled job)

Compares production feature values (FY2025) with the window the model was trained on
(FY2023-24), writes an HTML report and a JSON summary, and recommends retraining when
the share of drifted features passes the threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from ..paths import MONITORING, ROOT

DRIFT_SHARE_THRESHOLD = 0.3  # retrain when more than 30% of features drift


@dataclass
class DriftSummary:
    drifted_features: list[str]
    drift_share: float
    p_values: dict[str, float]
    retrain_recommended: bool
    report_html: str


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_SHARE_THRESHOLD,
    html_path: str | None = None,
) -> DriftSummary:
    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    p_values: dict[str, float] = {}
    for metric in snapshot.dict()["metrics"]:
        match = re.match(r"ValueDrift\(column=([^,]+),method=.*p_value,threshold=([\d.]+)\)", metric["metric_name"])
        if match:
            p_values[match.group(1)] = float(metric["value"])
    drifted = sorted(col for col, p in p_values.items() if p < 0.05)
    share = len(drifted) / len(p_values) if p_values else 0.0
    path = html_path or str(MONITORING / "drift_report.html")
    snapshot.save_html(path)
    shown = str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else path
    return DriftSummary(
        drifted, round(share, 3), {k: round(v, 6) for k, v in p_values.items()}, share > threshold, shown
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    summary = check_drift(
        pd.read_parquet(MONITORING / "reference.parquet"), pd.read_parquet(MONITORING / "current.parquet")
    )
    (MONITORING / "drift_summary.json").write_text(json.dumps(asdict(summary), indent=2))
    print(json.dumps(asdict(summary), indent=2))
    if summary.retrain_recommended:
        print("DRIFT DETECTED: retraining recommended")
        if args.fail_on_drift:
            sys.exit(1)


if __name__ == "__main__":
    main()
