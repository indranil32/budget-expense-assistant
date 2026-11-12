import numpy as np
import pandas as pd

from budget_assistant.experiments.bayesian_ab import VARIANTS, analyse, analyse_events, assign_variant, simulate_events
from budget_assistant.monitoring.drift import check_drift


def _frame(rng: np.random.Generator, shift: float = 0.0, n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({"a": rng.normal(1 + shift, 0.1, n), "b": rng.normal(0, 1, n), "c": rng.normal(5, 1, n)})


def test_no_drift_on_same_distribution(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    summary = check_drift(_frame(rng), _frame(rng), html_path=f"{tmp_path}/r.html")
    assert not summary.retrain_recommended


def test_drift_detected_on_shift(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    summary = check_drift(_frame(rng), _frame(rng, shift=0.3), html_path=f"{tmp_path}/r.html")
    assert summary.drifted_features == ["a"] and summary.retrain_recommended


def test_assignment_is_sticky_and_balanced() -> None:
    assert assign_variant("user-1") == assign_variant("user-1")
    share = sum(assign_variant(f"u{i}") == VARIANTS[1] for i in range(4000)) / 4000
    assert 0.45 < share < 0.55


def test_clear_winner_ships_and_tie_keeps_collecting() -> None:
    assert analyse(600, 1000, 700, 1000).decision == "ship B (guided)"
    tie = analyse(500, 1000, 505, 1000)
    assert tie.decision == "keep collecting data" and 0.3 < tie.prob_b_beats_a < 0.8


def test_simulated_experiment_recovers_the_better_flow() -> None:
    result = analyse_events(simulate_events(0.60, 0.70, users=3000))
    assert result.prob_b_beats_a > 0.99
