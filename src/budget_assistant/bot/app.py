"""Azure AI Bot Service front end, built on the Microsoft 365 Agents SDK (successor to the Bot Framework SDK).

    python -m budget_assistant.bot.app       # serves POST /api/messages on port 3978

Register an Azure Bot resource pointing at /api/messages, enable the Teams and Web Chat
channels, and set the CONNECTIONS__SERVICE_CONNECTION__SETTINGS__* variables (see
.env.example). For local testing use the Microsoft 365 Agents Playground.

The bot is a thin adapter: all journey logic lives in ``BudgetExpenseAssistant``, so it
can be tested without the channel and reused behind other front ends.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..assistant import GREETING, BudgetExpenseAssistant


async def handle_message(assistant: BudgetExpenseAssistant, context: Any) -> None:
    activity = context.activity
    reply = assistant.handle(
        conversation_id=activity.conversation.id,
        user_id=activity.from_property.id if activity.from_property else "anonymous",
        text=activity.text or "",
    )
    await context.send_activity(reply.text)


def create_agent_app(assistant: BudgetExpenseAssistant, env: Mapping[str, str] | None = None) -> tuple[Any, Any, Any]:
    from microsoft_agents.activity import load_configuration_from_env
    from microsoft_agents.authentication.msal import MsalConnectionManager
    from microsoft_agents.hosting.aiohttp import CloudAdapter
    from microsoft_agents.hosting.core import AgentApplication, MemoryStorage, TurnState

    config = load_configuration_from_env(env if env is not None else os.environ)
    connections = MsalConnectionManager(**config)
    adapter = CloudAdapter(connection_manager=connections)
    app = AgentApplication[TurnState](
        storage=MemoryStorage(), adapter=adapter, connection_manager=connections, **config
    )

    @app.conversation_update("membersAdded")
    async def on_members_added(context: Any, _state: Any) -> None:
        await context.send_activity("Hello! " + GREETING)

    @app.activity("message")
    async def on_message(context: Any, _state: Any) -> None:
        await handle_message(assistant, context)

    return app, adapter, connections


def build_assistant() -> BudgetExpenseAssistant:
    """Production wiring: Azure OpenAI chat, Azure AI Search hybrid retrieval, fine-tuned classifier if deployed."""
    import pandas as pd

    from ..classify import BaselineClassifier, fine_tuned_classifier
    from ..llm import AzureChat, AzureEmbedder
    from ..paths import CLAIMS_CSV
    from ..rag.retrieval import AzureSearchRetriever

    claims = pd.read_csv(CLAIMS_CSV)
    finetuned = os.environ.get("AZURE_OPENAI_FINETUNED_DEPLOYMENT")
    classifier: Any = (
        fine_tuned_classifier(AzureChat(finetuned))
        if finetuned
        else BaselineClassifier(claims[claims["split"] == "train"])
    )
    return BudgetExpenseAssistant(AzureChat(), AzureSearchRetriever(AzureEmbedder()), classifier)


def main() -> None:
    from aiohttp.web import Application, Request, Response, run_app
    from dotenv import load_dotenv
    from microsoft_agents.hosting.aiohttp import jwt_authorization_middleware, start_agent_process

    load_dotenv()
    agent_app, adapter, connections = create_agent_app(build_assistant())

    async def messages(request: Request) -> Response:
        response = await start_agent_process(request, agent_app, adapter)
        return response or Response(status=202)

    web = Application(middlewares=[jwt_authorization_middleware])
    web.router.add_post("/api/messages", messages)
    web["agent_configuration"] = connections.get_default_connection_configuration()
    run_app(web, host="0.0.0.0", port=int(os.environ.get("PORT", "3978")))


if __name__ == "__main__":
    main()
