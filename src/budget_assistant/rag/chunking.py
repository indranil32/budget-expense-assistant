"""Chunking strategies for policy documents. ``eval_retrieval`` compares them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..paths import POLICIES


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    section: str
    text: str


def load_documents(folder: Path = POLICIES) -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted(folder.glob("*.md"))}


def fixed_size(docs: dict[str, str], size: int = 300, overlap: int = 60) -> list[Chunk]:
    """Baseline: fixed character windows with overlap, ignoring document structure."""
    chunks = []
    for doc, text in docs.items():
        flat = re.sub(r"\s+", " ", text).strip()
        start, i = 0, 0
        while start < len(flat):
            chunks.append(Chunk(f"{doc}#w{i}", doc, "", flat[start : start + size]))
            start += size - overlap
            i += 1
    return chunks


def by_heading(docs: dict[str, str]) -> list[Chunk]:
    """Structure-aware: one chunk per '##' section, prefixed with the document title and
    section name so each chunk is self-describing (helps both keyword and vector search)."""
    chunks = []
    for doc, text in docs.items():
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else doc
        for i, part in enumerate(re.split(r"^## ", text, flags=re.M)[1:]):
            heading, _, body = part.partition("\n")
            body = re.sub(r"\s+", " ", body).strip()
            chunks.append(Chunk(f"{doc}#s{i}", doc, heading.strip(), f"{title} - {heading.strip()}: {body}"))
    return chunks


STRATEGIES = {"fixed_size": fixed_size, "by_heading": by_heading}
