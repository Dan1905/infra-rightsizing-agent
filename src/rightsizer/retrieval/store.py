"""Retrieval over the policy / runbook / postmortem corpus.

Markdown is split on headings so every chunk keeps a citable address
(`payment-service-runbook.md > Binding constraints`). Embeddings are computed
locally with sentence-transformers and stored in a persistent Chroma
collection; the model never sees the raw corpus, only the retrieved chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MAX_CHUNK_CHARS = 1400


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str
    distance: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "citation": self.citation,
            "source": self.source,
            "heading": self.heading,
            "distance": self.distance,
            "text": self.text,
        }


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split an oversized section on paragraph boundaries."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) + 2 > limit:
            parts.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(path: Path, root: Path) -> list[Chunk]:
    """One chunk per heading section, carrying the heading path as its address."""
    source = str(path.relative_to(root))
    raw = path.read_text(encoding="utf-8")

    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    current_heading = ""
    body: list[str] = []

    for line in raw.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if body and any(b.strip() for b in body):
                sections.append((current_heading, body))
            level, title = len(match.group(1)), match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = " > ".join(heading_stack[1:]) or title
            body = []
        else:
            body.append(line)
    if body and any(b.strip() for b in body):
        sections.append((current_heading, body))

    chunks: list[Chunk] = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        for i, piece in enumerate(_split_long(text)):
            # Prefix the address so the embedding sees which document it is from.
            embedded = f"[{source} > {heading}]\n{piece}" if heading else f"[{source}]\n{piece}"
            digest = hashlib.sha1(f"{source}:{heading}:{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(id=digest, text=embedded, source=source, heading=heading))
    return chunks


def load_corpus(policies_dir: Path) -> list[Chunk]:
    files = sorted(policies_dir.rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown documents under {policies_dir}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_markdown(path, policies_dir))
    return chunks


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=2)
def _embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    # Prefer the local cache. Otherwise the library checks the Hugging Face Hub
    # on every load, which stalls or fails on a slow or offline connection.
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def _embed(model_name: str, texts: Iterable[str]) -> list[list[float]]:
    model = _embedder(model_name)
    return model.encode(list(texts), normalize_embeddings=True).tolist()


class PolicyStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, settings: Settings):
        import chromadb

        self.settings = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def index(self, rebuild: bool = False) -> int:
        """(Re)build the index. Returns the number of chunks stored."""
        import chromadb

        if rebuild:
            self._client.delete_collection(self.settings.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        chunks = load_corpus(self.settings.policies_dir)
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=_embed(self.settings.embedding_model, (c.text for c in chunks)),
            metadatas=[{"source": c.source, "heading": c.heading} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or self.settings.top_k
        if self.count() == 0:
            raise RuntimeError(
                "Policy index is empty. Run `rightsize index` first."
            )
        result = self._collection.query(
            query_embeddings=_embed(self.settings.embedding_model, [query]),
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[Chunk] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append(
                Chunk(
                    id=cid,
                    text=doc,
                    source=str(meta.get("source", "")),
                    heading=str(meta.get("heading", "")),
                    distance=float(dist),
                )
            )
        return chunks


def dedupe(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Keep the best-scoring copy of each chunk across multiple queries."""
    best: dict[str, Chunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.id)
        if existing is None or (chunk.distance or 0) < (existing.distance or 0):
            best[chunk.id] = chunk
    return sorted(best.values(), key=lambda c: c.distance if c.distance is not None else 1.0)


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"--- [{i + 1}] {c.citation} (distance {c.distance:.3f})\n{c.text}"
        if c.distance is not None
        else f"--- [{i + 1}] {c.citation}\n{c.text}"
        for i, c in enumerate(chunks)
    )
