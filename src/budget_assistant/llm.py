"""Model access: Azure OpenAI / Foundry deployments with Entra ID auth, plus offline stand-ins.

Online:
  AzureChat       chat completions with JSON-schema structured output
  AzureEmbedder   text embeddings for vector search
Offline (tests and CI, no credentials):
  HashingEmbedder deterministic feature-hashing vectors — a stand-in, not a semantic model
  ScriptedChat    replays canned responses
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class Chat(Protocol):
    def complete(self, system: str, user: str, schema: type[BaseModel] | None = None) -> ChatResult: ...


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""

    def parse(self, schema: type[BaseModel]) -> Any:
        return schema.model_validate_json(self.text)


def _azure_client() -> Any:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


class AzureChat:
    def __init__(self, deployment: str | None = None, client: Any = None) -> None:
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-model")
        self.client = client or _azure_client()

    def complete(self, system: str, user: str, schema: type[BaseModel] | None = None) -> ChatResult:
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
            }
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            **kwargs,
        )
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.deployment,
        )


class AzureEmbedder:
    def __init__(self, deployment: str | None = None, client: Any = None) -> None:
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding-model")
        self.client = client or _azure_client()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.deployment, input=list(texts))
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class HashingEmbedder:
    """Offline stand-in for an embedding model: character n-gram feature hashing.

    It captures surface similarity only. Use AzureEmbedder for real semantic retrieval;
    this exists so retrieval code and evals run in CI without credentials.
    """

    def __init__(self, dims: int = 1024) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dims = dims
        self._vec = HashingVectorizer(n_features=dims, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._vec.transform([t.lower() for t in texts]).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)


@dataclass
class ScriptedChat:
    """Replays canned responses in order (tests)."""

    responses: list[str | dict[str, Any]]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str, schema: type[BaseModel] | None = None) -> ChatResult:
        self.calls.append((system, user))
        item = self.responses.pop(0)
        text = json.dumps(item) if isinstance(item, dict) else item
        return ChatResult(
            text=text, input_tokens=len(system + user) // 4, output_tokens=len(text) // 4, model="scripted"
        )
