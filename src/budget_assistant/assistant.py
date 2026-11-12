"""Multi-journey conversational assistant for budget and expense questions.

Journeys
  policy_question  RAG over budget circulars and expense policies; answers must cite
                   sources, otherwise the assistant declines (grounding guardrail).
  submit_claim     multi-turn slot filling, deterministic policy checks, category
                   classification. Two flows run as an A/B experiment:
                     A single_prompt - extract everything from one message
                     B guided        - ask for one field at a time
  claim_status     look up a claim by id.

The LLM classifies intent, extracts fields and writes grounded answers. Business rules
(approval levels, receipts, deadlines) are plain code, never left to the model.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from .classify import Classifier
from .experiments.bayesian_ab import EVENTS_PATH, assign_variant
from .llm import Chat
from .rag.chunking import Chunk

Intent = Literal["policy_question", "submit_claim", "claim_status", "cancel", "other"]
CLAIM_ID = re.compile(r"\bC-\d{5}\b", re.I)
GREETING = (
    "I can answer budget and expense policy questions, submit an expense claim, "
    "or check a claim's status (e.g. 'status of C-00042')."
)


class IntentDecision(BaseModel):
    intent: Intent


class GroundedAnswer(BaseModel):
    answerable: bool = Field(description="False if the sources do not contain the answer")
    answer: str
    citations: list[int] = Field(description="Numbers of the sources used, e.g. [1, 3]")


class ClaimDraft(BaseModel):
    description: str | None = None
    amount_aed: float | None = None
    expense_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")


class Retriever(Protocol):  # LocalHybridRetriever or AzureSearchRetriever
    def search(self, query: str, k: int = 3) -> list[Chunk]: ...


@dataclass
class Reply:
    text: str
    journey: str
    sources: list[str] = field(default_factory=list)


@dataclass
class ClaimRecord:
    claim_id: str
    description: str
    amount_aed: float
    expense_date: str
    category: str
    approver: str
    status: str = "submitted"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    user_id: str
    journey: str | None = None
    variant: str = "single_prompt"
    draft: ClaimDraft = field(default_factory=ClaimDraft)


# ---- deterministic policy rules (Circular FIN-11) --------------------------------


def approver_for(amount: float) -> str:
    if amount <= 5_000:
        return "line manager"
    if amount <= 25_000:
        return "department director"
    return "finance director"


def policy_warnings(amount: float, expense_date: dt.date, today: dt.date) -> list[str]:
    warnings = []
    if amount > 100:
        warnings.append("An itemised receipt or tax invoice is required (amount above AED 100).")
    if (today - expense_date).days > 30:
        warnings.append("Submitted more than 30 days after the expense: director approval is required.")
    if expense_date > today:
        warnings.append("The expense date is in the future; please check it.")
    return warnings


class ClaimStore:
    def __init__(self) -> None:
        self.claims: dict[str, ClaimRecord] = {}

    def add(self, record: ClaimRecord) -> ClaimRecord:
        self.claims[record.claim_id] = record
        return record

    def next_id(self) -> str:
        return f"C-{90000 + len(self.claims):05d}"


# ---- the assistant ---------------------------------------------------------------


class BudgetExpenseAssistant:
    def __init__(
        self,
        chat: Chat,
        retriever: Retriever,
        classifier: Classifier,
        store: ClaimStore | None = None,
        today: dt.date | None = None,
        events_path: Path | None = EVENTS_PATH,
    ) -> None:
        self.chat, self.retriever, self.classifier = chat, retriever, classifier
        self.store = store or ClaimStore()
        self.today = today or dt.date.today()
        self.events_path = events_path
        self.states: dict[str, ConversationState] = {}

    def handle(self, conversation_id: str, user_id: str, text: str) -> Reply:
        state = self.states.setdefault(conversation_id, ConversationState(user_id, variant=assign_variant(user_id)))
        intent = self._intent(text, state)
        if intent == "cancel":
            if state.journey == "submit_claim":
                self._log_outcome(state, completed=False)
            self.states.pop(conversation_id)
            return Reply("Cancelled. " + GREETING, "cancel")
        if state.journey == "submit_claim" and intent in ("submit_claim", "other"):
            return self._continue_claim(state, text, conversation_id)
        if state.journey == "submit_claim":  # switched journeys mid-claim -> abandoned
            self._log_outcome(state, completed=False)
            state.journey, state.draft = None, ClaimDraft()
        if intent == "policy_question":
            return self._policy_answer(text)
        if intent == "claim_status":
            return self._claim_status(text)
        if intent == "submit_claim":
            state.journey = "submit_claim"
            return self._continue_claim(state, text, conversation_id)
        return Reply(GREETING, "other")

    # intent: a cheap regex catches claim ids; everything else goes to the model
    def _intent(self, text: str, state: ConversationState) -> Intent:
        if text.strip().lower() in ("cancel", "stop", "never mind"):
            return "cancel"
        if CLAIM_ID.search(text) and state.journey != "submit_claim":
            return "claim_status"
        context = "(The user is in the middle of submitting a claim.) " if state.journey else ""
        result = self.chat.complete(
            "Classify the user's message for a government budget & expense assistant. "
            "policy_question: asks what the rules are. submit_claim: wants to file an expense or is giving claim "
            "details. claim_status: asks about an existing claim. cancel: wants to stop. other: anything else.",
            context + text,
            schema=IntentDecision,
        )
        try:
            return result.parse(IntentDecision).intent  # type: ignore[no-any-return]
        except ValueError:
            return "other"

    def _policy_answer(self, question: str) -> Reply:
        chunks = self.retriever.search(question, k=3)
        sources = "\n".join(f"[{i}] ({c.doc} / {c.section or 'excerpt'}) {c.text}" for i, c in enumerate(chunks, 1))
        result = self.chat.complete(
            "Answer using ONLY the numbered sources. Cite the source numbers you used. "
            "If the sources do not answer the question, set answerable=false.",
            f"Sources:\n{sources}\n\nQuestion: {question}",
            schema=GroundedAnswer,
        )
        try:
            answer = result.parse(GroundedAnswer)
        except ValueError:
            answer = GroundedAnswer(answerable=False, answer="", citations=[])
        cited = [chunks[i - 1] for i in answer.citations if 1 <= i <= len(chunks)]
        if not answer.answerable or not cited:  # grounding guardrail: no citation, no answer
            return Reply(
                "I couldn't find that in the budget and expense policies. Please contact the finance helpdesk.",
                "policy_question",
            )
        refs = sorted({f"{c.doc} / {c.section}" if c.section else c.doc for c in cited})
        return Reply(f"{answer.answer}\n\nSources: " + "; ".join(refs), "policy_question", refs)

    def _claim_status(self, text: str) -> Reply:
        match = CLAIM_ID.search(text)
        record = self.store.claims.get(match.group(0).upper()) if match else None
        if record is None:
            return Reply("I couldn't find that claim. Claim ids look like C-90001.", "claim_status")
        return Reply(
            f"{record.claim_id}: {record.status}, AED {record.amount_aed:,.2f}, awaiting the {record.approver}.",
            "claim_status",
        )

    def _continue_claim(self, state: ConversationState, text: str, conversation_id: str) -> Reply:
        if state.variant == "single_prompt" or state.draft.description is None:
            extracted = self.chat.complete(
                f"Extract expense-claim fields from the message. Today is {self.today.isoformat()}. "
                "Leave a field null if it is not stated.",
                text,
                schema=ClaimDraft,
            )
            try:
                found = extracted.parse(ClaimDraft)
            except ValueError:
                found = ClaimDraft()
            for name in ("description", "amount_aed", "expense_date"):
                if getattr(state.draft, name) is None and getattr(found, name) is not None:
                    setattr(state.draft, name, getattr(found, name))
        else:  # guided flow: the reply answers the one field we asked for
            self._fill_next_field(state.draft, text)

        missing = [n for n in ("description", "amount_aed", "expense_date") if getattr(state.draft, n) is None]
        if missing:
            labels = {
                "description": "what the expense was for",
                "amount_aed": "the amount in AED",
                "expense_date": "the date of the expense (YYYY-MM-DD)",
            }
            if state.variant == "guided":
                return Reply(f"Please tell me {labels[missing[0]]}.", "submit_claim")
            return Reply("I still need " + ", ".join(labels[m] for m in missing) + ".", "submit_claim")
        return self._submit(state, conversation_id)

    @staticmethod
    def _fill_next_field(draft: ClaimDraft, text: str) -> None:
        if draft.amount_aed is None:
            number = re.search(r"\d[\d,]*(\.\d+)?", text)
            if number:
                draft.amount_aed = float(number.group(0).replace(",", ""))
        elif draft.expense_date is None:
            date = re.search(r"\d{4}-\d{2}-\d{2}", text)
            if date:
                draft.expense_date = date.group(0)

    def _submit(self, state: ConversationState, conversation_id: str) -> Reply:
        draft = state.draft
        assert draft.description and draft.amount_aed is not None and draft.expense_date
        try:
            expense_date = dt.date.fromisoformat(draft.expense_date)
        except ValueError:
            draft.expense_date = None
            return Reply("That date doesn't look right. Please use YYYY-MM-DD.", "submit_claim")
        record = self.store.add(
            ClaimRecord(
                claim_id=self.store.next_id(),
                description=draft.description,
                amount_aed=draft.amount_aed,
                expense_date=expense_date.isoformat(),
                category=self.classifier.predict(draft.description).category,
                approver=approver_for(draft.amount_aed),
                warnings=policy_warnings(draft.amount_aed, expense_date, self.today),
            )
        )
        self._log_outcome(state, completed=True)
        self.states.pop(conversation_id, None)
        notes = "".join(f"\n- {w}" for w in record.warnings)
        return Reply(
            f"Submitted {record.claim_id}: {record.description} (category: {record.category}), "
            f"AED {record.amount_aed:,.2f}. It goes to the {record.approver} for approval.{notes}",
            "submit_claim",
        )

    def _log_outcome(self, state: ConversationState, completed: bool) -> None:
        if self.events_path is None:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "user_id": state.user_id,
                        "variant": state.variant,
                        "completed": completed,
                        "at": dt.datetime.now(dt.UTC).isoformat(),
                    }
                )
                + "\n"
            )
