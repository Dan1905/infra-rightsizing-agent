"""Shared fakes. Nothing here touches Docker, a cluster, the network or a model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rightsizer.audit import AuditLog
from rightsizer.backends.base import MIB, WorkloadMetrics
from rightsizer.config import Settings
from rightsizer.retrieval.store import Chunk


def make_metrics(name: str = "web-frontend", **overrides) -> WorkloadMetrics:
    base = WorkloadMetrics(
        name=name,
        kind="Deployment",
        status="available",
        tier="frontend",
        managed=True,
        restart_count=0,
        restart_scope="window",
        mem_max_bytes=5 * MIB,
        cpu_max_cores=0.08,
    )
    return replace(base, **overrides)


def chunk(source: str, heading: str, text: str = "body", distance: float = 0.3) -> Chunk:
    return Chunk(
        id=f"{source}:{heading}",
        text=f"[{source} > {heading}]\n{text}",
        source=source,
        heading=heading,
        distance=distance,
    )


class FakeStore:
    """Stands in for PolicyStore: returns canned passages, records queries."""

    def __init__(self, chunks: list[Chunk], top_k: int = 3):
        self.chunks = chunks
        self.settings = Settings(top_k=top_k)
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        self.queries.append(query)
        return self.chunks[: k or self.settings.top_k]


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        chunk("workload-profiles.md", "web-frontend"),
        chunk("workload-profiles.md", "batch-worker"),
        chunk("payment-service-runbook.md", "Binding constraints"),
        chunk("resource-sizing-policy.md", "Headroom requirements"),
    ]


@pytest.fixture
def store(corpus) -> FakeStore:
    return FakeStore(corpus, top_k=4)


@pytest.fixture
def audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.db")


@pytest.fixture
def workloads() -> list[WorkloadMetrics]:
    return [
        make_metrics("web-frontend"),
        make_metrics("batch-worker", tier="batch", min_window="24h"),
        make_metrics("payment-service", tier="payments", restart_count=7),
        make_metrics("prometheus", managed=False, tier=None),
    ]
