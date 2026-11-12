"""Release gate.

    python evals/run_evals.py           # offline (CI): retrieval quality + classifier floor
    python evals/run_evals.py --live    # + grounded policy answers from the Azure deployment

Live checks: each in-scope answer must cite the expected policy and contain the key fact;
each out-of-scope question must be refused (no invented policy). Exits 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from budget_assistant.classify import BaselineClassifier
from budget_assistant.finetune.benchmark import run as run_classifier
from budget_assistant.finetune.prepare import split
from budget_assistant.llm import HashingEmbedder
from budget_assistant.paths import RESULTS
from budget_assistant.rag.eval_retrieval import evaluate

HERE = Path(__file__).parent
GATES = {"retrieval_hit@3": 0.85, "classifier_accuracy": 0.80, "policy_answer_accuracy": 0.8, "refusal_accuracy": 1.0}


def live_policy_eval() -> dict[str, float]:
    from budget_assistant.assistant import BudgetExpenseAssistant
    from budget_assistant.llm import AzureChat, AzureEmbedder
    from budget_assistant.rag.chunking import by_heading, load_documents
    from budget_assistant.rag.retrieval import LocalHybridRetriever

    train, _, _ = split()
    assistant = BudgetExpenseAssistant(
        AzureChat(),
        LocalHybridRetriever(by_heading(load_documents()), AzureEmbedder()),
        BaselineClassifier(train),
        events_path=None,
    )
    cases = [json.loads(line) for line in (HERE / "policy_qa_golden.jsonl").read_text().splitlines() if line.strip()]
    answered = refused = n_answer = n_refuse = 0
    for i, case in enumerate(cases):
        reply = assistant.handle(f"eval-{i}", "eval", case["q"])
        if case.get("expect_refusal"):
            n_refuse += 1
            refused += not reply.sources
        else:
            n_answer += 1
            answered += (
                any(case["expect_doc"] in s for s in reply.sources)
                and case["expect_text"].lower() in reply.text.lower()
            )
    return {"policy_answer_accuracy": answered / n_answer, "refusal_accuracy": refused / n_refuse}


def main(live: bool) -> int:
    train, _, test = split()
    report: dict[str, Any] = {
        "retrieval_hit@3": evaluate(HashingEmbedder())["by_heading/hybrid"]["hit@3"],
        "classifier_accuracy": run_classifier(BaselineClassifier(train), test, 0.0, 0.0)["accuracy"],
    }
    if live:
        report.update(live_policy_eval())
    failures = [k for k, floor in GATES.items() if k in report and report[k] < floor]
    report["failures"] = failures
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"release_gate_{'live' if live else 'offline'}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("RELEASE GATE:", "FAIL " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    sys.exit(main(parser.parse_args().live))
