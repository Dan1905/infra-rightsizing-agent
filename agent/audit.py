"""Append-only audit trail in SQLite.

Everything that would be needed to reconstruct why a change happened:
the run, every retrieval the model performed, each proposal with the policy
it cited, the human's typed decision, and the execution result.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    model           TEXT NOT NULL,
    lookback        TEXT NOT NULL,
    prometheus_url  TEXT NOT NULL,
    container_count INTEGER DEFAULT 0,
    proposal_count  INTEGER DEFAULT 0,
    transcript_json TEXT
);

CREATE TABLE IF NOT EXISTS retrievals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    created_at   TEXT NOT NULL,
    origin       TEXT NOT NULL,          -- 'seed' or 'tool'
    query        TEXT NOT NULL,
    results_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    created_at        TEXT NOT NULL,
    container         TEXT NOT NULL,
    action            TEXT NOT NULL,
    params_json       TEXT NOT NULL,
    reason            TEXT NOT NULL,
    policy_cited_json TEXT NOT NULL,
    confidence        TEXT,
    estimated_saving  TEXT,
    context_json      TEXT NOT NULL,     -- the chunks that grounded this proposal
    approval          TEXT,              -- 'yes' | 'no' | 'skipped'
    decided_at        TEXT,
    executed          INTEGER DEFAULT 0,
    exec_result       TEXT,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_run ON retrievals(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- runs ---------------------------------------------------------------

    def start_run(self, *, model: str, lookback: str, prometheus_url: str) -> str:
        run_id = f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, model, lookback, prometheus_url)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, _now(), model, lookback, prometheus_url),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        container_count: int,
        proposal_count: int,
        transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, container_count = ?, proposal_count = ?,"
                " transcript_json = ? WHERE run_id = ?",
                (
                    _now(),
                    container_count,
                    proposal_count,
                    json.dumps(transcript or [], default=str),
                    run_id,
                ),
            )

    # -- retrievals ---------------------------------------------------------

    def log_retrieval(
        self, run_id: str, *, origin: str, query: str, results: list[dict[str, Any]]
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO retrievals (run_id, created_at, origin, query, results_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, _now(), origin, query, json.dumps(results, default=str)),
            )

    # -- decisions ----------------------------------------------------------

    def log_proposal(
        self,
        run_id: str,
        *,
        container: str,
        action: str,
        params: dict[str, Any],
        reason: str,
        policy_cited: list[str],
        confidence: str | None,
        estimated_saving: str | None,
        context: list[dict[str, Any]],
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO decisions (run_id, created_at, container, action, params_json,"
                " reason, policy_cited_json, confidence, estimated_saving, context_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _now(),
                    container,
                    action,
                    json.dumps(params, default=str),
                    reason,
                    json.dumps(policy_cited),
                    confidence,
                    estimated_saving,
                    json.dumps(context, default=str),
                ),
            )
            return int(cur.lastrowid)

    def record_decision(self, decision_id: int, approval: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET approval = ?, decided_at = ? WHERE id = ?",
                (approval, _now(), decision_id),
            )

    def record_execution(
        self, decision_id: int, *, ok: bool, result: str | None, error: str | None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET executed = ?, exec_result = ?, error = ? WHERE id = ?",
                (1 if ok else 0, result, error, decision_id),
            )

    # -- reading ------------------------------------------------------------

    def recent_decisions(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            )

    def run_summary(self, limit: int = 10) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
            )
