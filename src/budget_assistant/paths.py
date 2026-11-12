"""Repository paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POLICIES = DATA / "policies"
CLAIMS_CSV = DATA / "claims.csv"
SPEND_CSV = DATA / "spend_monthly.csv"
FEAST_DATA = DATA / "feast"
FEATURE_REPO = ROOT / "feature_repo"
MONITORING = DATA / "monitoring"
RESULTS = ROOT / "results"
