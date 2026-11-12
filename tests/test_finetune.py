import json

from budget_assistant.classify import FINE_TUNED_SYSTEM, BaselineClassifier, ClaimCategory
from budget_assistant.finetune.benchmark import run
from budget_assistant.finetune.prepare import split, to_chat_record


def test_chat_record_format() -> None:
    record = to_chat_record("USB-C docking station", "it_equipment")
    roles = [m["role"] for m in record["messages"]]  # type: ignore[index]
    assert roles == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == FINE_TUNED_SYSTEM  # type: ignore[index]
    ClaimCategory.model_validate_json(record["messages"][2]["content"])  # type: ignore[index]


def test_test_set_contains_unseen_wording_and_no_overlap() -> None:
    train, _valid, test = split()
    assert set(train["claim_id"]).isdisjoint(test["claim_id"])
    unseen = set(test["description"].str.lower()) - set(train["description"].str.lower())
    assert len(unseen) / len(test) > 0.3


def test_benchmark_reports_quality_latency_and_cost() -> None:
    train, _, test = split()
    row = run(BaselineClassifier(train), test, price_in=0.0, price_out=0.0)
    assert set(row) == {"accuracy", "macro_f1", "mean_latency_ms", "cost_per_1k_claims_usd"}
    assert row["accuracy"] > 0.7
    json.dumps(row)
