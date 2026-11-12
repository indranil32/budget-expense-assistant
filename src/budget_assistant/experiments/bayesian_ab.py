"""Bayesian A/B testing of conversational flows (Beta-Binomial).

    python -m budget_assistant.experiments.bayesian_ab                     # analyse logged events
    python -m budget_assistant.experiments.bayesian_ab --simulate 0.62 0.68 # simulate traffic, then analyse

Experiment: the expense-claim journey has two flows.
  A = "single_prompt": user describes the whole claim in one message.
  B = "guided":        the assistant asks for each field step by step.
Metric: task completion (claim submitted without abandoning).

Why Bayesian: it answers the question stakeholders actually ask ("how likely is B
better, and what do we risk if we ship it?") and supports decisions at any sample size
without the peeking problem of repeated significance tests.

Decision rule: ship B when P(B > A) >= 95% AND the expected loss of choosing B is
below 0.2 percentage points; otherwise keep collecting data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..paths import RESULTS

VARIANTS = ("single_prompt", "guided")
EVENTS_PATH = RESULTS / "ab_events.jsonl"
PROB_THRESHOLD = 0.95
LOSS_THRESHOLD = 0.002


def assign_variant(user_id: str, experiment: str = "claim_flow_v1") -> str:
    """Deterministic 50/50 split: the same user always sees the same flow."""
    bucket = int(hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest(), 16) % 100
    return VARIANTS[0] if bucket < 50 else VARIANTS[1]


@dataclass
class ABResult:
    n_a: int
    conversions_a: int
    n_b: int
    conversions_b: int
    posterior_mean_a: float
    posterior_mean_b: float
    prob_b_beats_a: float
    expected_loss_choose_b: float
    expected_loss_choose_a: float
    lift_ci95: tuple[float, float]
    decision: str


def analyse(
    conv_a: int,
    n_a: int,
    conv_b: int,
    n_b: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 200_000,
    seed: int = 0,
) -> ABResult:
    rng = np.random.default_rng(seed)
    a = rng.beta(prior[0] + conv_a, prior[1] + n_a - conv_a, draws)
    b = rng.beta(prior[0] + conv_b, prior[1] + n_b - conv_b, draws)
    prob_b = float((b > a).mean())
    loss_b = float(np.maximum(a - b, 0).mean())  # completion rate we give up, on average, if B is actually worse
    loss_a = float(np.maximum(b - a, 0).mean())
    lift = b - a
    if prob_b >= PROB_THRESHOLD and loss_b < LOSS_THRESHOLD:
        decision = "ship B (guided)"
    elif (1 - prob_b) >= PROB_THRESHOLD and loss_a < LOSS_THRESHOLD:
        decision = "keep A (single_prompt)"
    else:
        decision = "keep collecting data"
    return ABResult(
        n_a,
        conv_a,
        n_b,
        conv_b,
        round(float(a.mean()), 4),
        round(float(b.mean()), 4),
        round(prob_b, 4),
        round(loss_b, 5),
        round(loss_a, 5),
        (round(float(np.quantile(lift, 0.025)), 4), round(float(np.quantile(lift, 0.975)), 4)),
        decision,
    )


def simulate_events(rate_a: float, rate_b: float, users: int = 2000, seed: int = 1) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    events = []
    for i in range(users):
        user = f"user-{i:05d}"
        variant = assign_variant(user)
        rate = rate_a if variant == VARIANTS[0] else rate_b
        events.append({"user_id": user, "variant": variant, "completed": bool(rng.random() < rate), "simulated": True})
    return events


def analyse_events(events: list[dict[str, object]]) -> ABResult:
    counts = {v: [0, 0] for v in VARIANTS}
    for e in events:
        counts[str(e["variant"])][0] += int(bool(e["completed"]))
        counts[str(e["variant"])][1] += 1
    (ca, na), (cb, nb) = counts[VARIANTS[0]], counts[VARIANTS[1]]
    return analyse(ca, na, cb, nb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", nargs=2, type=float, metavar=("RATE_A", "RATE_B"))
    parser.add_argument("--users", type=int, default=2000)
    args = parser.parse_args()
    if args.simulate:
        events = simulate_events(args.simulate[0], args.simulate[1], users=args.users)
        print(f"Simulated {len(events)} users (true rates A={args.simulate[0]}, B={args.simulate[1]})")
    else:
        events = [json.loads(line) for line in Path(EVENTS_PATH).read_text().splitlines() if line.strip()]
    result = analyse_events(events)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ab_result.json").write_text(json.dumps(asdict(result), indent=2))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
