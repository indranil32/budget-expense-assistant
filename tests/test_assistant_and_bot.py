from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import TODAY, user_in_variant

from budget_assistant.assistant import BudgetExpenseAssistant, approver_for
from budget_assistant.bot.app import create_agent_app, handle_message
from budget_assistant.classify import BaselineClassifier
from budget_assistant.llm import ScriptedChat
from budget_assistant.rag.retrieval import LocalHybridRetriever


def make(
    chat: ScriptedChat, retriever: LocalHybridRetriever, classifier: BaselineClassifier, events: Path
) -> BudgetExpenseAssistant:
    return BudgetExpenseAssistant(chat, retriever, classifier, today=TODAY, events_path=events)


def events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_policy_answer_cites_sources(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    chat = ScriptedChat(
        [
            {"intent": "policy_question"},
            {"answerable": True, "answer": "Claims above AED 5,000 go to the department director.", "citations": [1]},
        ]
    )
    reply = make(chat, retriever, classifier, tmp_path / "e.jsonl").handle("c1", "u1", "Who approves AED 12,000?")
    assert reply.journey == "policy_question" and reply.sources
    assert "Sources:" in reply.text
    assert "[1]" in chat.calls[1][1]  # the model saw numbered sources


def test_ungrounded_answer_is_refused(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    chat = ScriptedChat(
        [{"intent": "policy_question"}, {"answerable": True, "answer": "You get a bonus.", "citations": []}]
    )
    reply = make(chat, retriever, classifier, tmp_path / "e.jsonl").handle("c1", "u1", "Do I get a bonus?")
    assert "couldn't find" in reply.text and not reply.sources


def test_single_prompt_claim_flow(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = tmp_path / "e.jsonl"
    user = user_in_variant("single_prompt")
    chat = ScriptedChat(
        [
            {"intent": "submit_claim"},
            {"description": "hotel stay in Riyadh, 2 nights", "amount_aed": 7200, "expense_date": "2026-09-01"},
        ]
    )
    assistant = make(chat, retriever, classifier, log)
    reply = assistant.handle("c1", user, "Claim AED 7,200 for my Riyadh hotel on 1 Sep 2026")
    assert reply.text.startswith("Submitted C-90000")
    record = assistant.store.claims["C-90000"]
    assert record.category == "travel" and record.approver == "department director"
    assert any("receipt" in w for w in record.warnings) and any("30 days" in w for w in record.warnings)
    assert events(log) == [{**events(log)[0], "variant": "single_prompt", "completed": True}]


def test_guided_claim_flow_asks_one_field_at_a_time(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    user = user_in_variant("guided")
    chat = ScriptedChat(
        [
            {"intent": "submit_claim"},
            {"description": "printer paper and toner"},
            {"intent": "submit_claim"},
            {"intent": "submit_claim"},
        ]
    )
    assistant = make(chat, retriever, classifier, tmp_path / "e.jsonl")
    assert "amount" in assistant.handle("c1", user, "I want to claim printer paper and toner").text
    assert "date" in assistant.handle("c1", user, "350").text
    done = assistant.handle("c1", user, "2026-11-02")
    assert "Submitted" in done.text and "office_supplies" in done.text and "line manager" in done.text


def test_cancel_logs_abandonment_and_status_lookup(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = tmp_path / "e.jsonl"
    chat = ScriptedChat([{"intent": "submit_claim"}, {"description": None}])
    assistant = make(chat, retriever, classifier, log)
    assistant.handle("c1", "u9", "new claim")
    assert assistant.handle("c1", "u9", "cancel").journey == "cancel"
    assert events(log)[0]["completed"] is False
    assert "couldn't find" in assistant.handle("c2", "u9", "status of C-12345?").text


@pytest.mark.parametrize(
    ("amount", "approver"), [(5000, "line manager"), (5000.01, "department director"), (25001, "finance director")]
)
def test_approval_levels(amount: float, approver: str) -> None:
    assert approver_for(amount) == approver


@pytest.mark.asyncio
async def test_bot_adapter_routes_messages_to_assistant(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sent: list[str] = []

    async def send_activity(text: str) -> None:
        sent.append(text)

    chat = ScriptedChat([{"intent": "other"}])
    context = SimpleNamespace(
        activity=SimpleNamespace(
            conversation=SimpleNamespace(id="conv"), from_property=SimpleNamespace(id="u1"), text="hi"
        ),
        send_activity=send_activity,
    )
    await handle_message(make(chat, retriever, classifier, tmp_path / "e.jsonl"), context)
    assert sent and "expense claim" in sent[0]


def test_agent_app_builds_with_service_connection(retriever, classifier, tmp_path) -> None:  # type: ignore[no-untyped-def]
    env = {
        f"CONNECTIONS__SERVICE_CONNECTION__SETTINGS__{k}": v
        for k, v in {
            "CLIENTID": "00000000-0000-0000-0000-000000000000",
            "CLIENTSECRET": "test",
            "TENANTID": "00000000-0000-0000-0000-000000000000",
        }.items()
    }
    app, _adapter, connections = create_agent_app(make(ScriptedChat([]), retriever, classifier, None), env)  # type: ignore[arg-type]
    assert app is not None and connections.get_default_connection_configuration() is not None
