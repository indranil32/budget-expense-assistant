"""Create the Azure AI Search index (keyword + HNSW vector) and upload policy chunks.

    python -m budget_assistant.rag.index_azure

Needs AZURE_SEARCH_ENDPOINT, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT and an
identity with "Search Index Data Contributor" + "Search Service Contributor" on the service.
"""

from __future__ import annotations

import os

from ..llm import AzureEmbedder, Embedder
from .chunking import Chunk, by_heading, load_documents


def build_index_definition(name: str, dims: int):  # type: ignore[no-untyped-def]
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.STRING, key=True),
            SimpleField(name="doc", type=SearchFieldDataType.STRING, filterable=True),
            SearchableField(name="section"),
            SearchableField(name="content"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.SINGLE),  # type: ignore[operator]
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="hnsw-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
        ),
    )


def documents_for_upload(chunks: list[Chunk], embedder: Embedder) -> list[dict[str, object]]:
    vectors = embedder.embed([c.text for c in chunks])
    return [
        {"id": c.id.replace("#", "-"), "doc": c.doc, "section": c.section, "content": c.text, "embedding": v.tolist()}
        for c, v in zip(chunks, vectors, strict=True)
    ]


def main() -> None:
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    name = os.environ.get("AZURE_SEARCH_INDEX", "policies")
    credential = DefaultAzureCredential()
    embedder = AzureEmbedder()
    docs = documents_for_upload(by_heading(load_documents()), embedder)
    dims = len(docs[0]["embedding"])  # type: ignore[arg-type]
    SearchIndexClient(endpoint, credential).create_or_update_index(build_index_definition(name, dims))
    SearchClient(endpoint, name, credential).upload_documents(docs)
    print(f"Indexed {len(docs)} chunks into '{name}' ({dims}-dim vectors)")


if __name__ == "__main__":
    main()
