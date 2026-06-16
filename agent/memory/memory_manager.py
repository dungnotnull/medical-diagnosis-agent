"""SQLite-based session memory manager with audit trail."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS diagnosis_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    triage_severity TEXT,
    news2_score INTEGER,
    qsofa_score INTEGER,
    curb65_score INTEGER,
    gcs_total INTEGER,
    body_system TEXT,
    red_flags TEXT,
    escalation_required INTEGER DEFAULT 0,
    top_icd_code TEXT,
    top_condition_name TEXT,
    patient_report_length INTEGER,
    safety_compliant INTEGER DEFAULT 1,
    llm_provider_used TEXT,
    session_data_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    use_case TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_hashes (
    hash TEXT PRIMARY KEY,
    source_url TEXT,
    title TEXT,
    added_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_severity ON diagnosis_sessions(triage_severity);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON diagnosis_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_cost_provider ON llm_cost_log(provider);
"""


class MemoryManager:
    def __init__(self, db_path: str = "data/medical_agent.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(DDL)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def save_session(
        self,
        session_id: str,
        triage_severity: str,
        news2_score: int,
        qsofa_score: int,
        curb65_score: int,
        gcs_total: Optional[int],
        body_system: str,
        red_flags: list[str],
        escalation_required: bool,
        top_icd_code: str,
        top_condition_name: str,
        patient_report_length: int,
        safety_compliant: bool,
        llm_provider_used: str,
        session_data: dict,
    ) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO diagnosis_sessions (
                        session_id, created_at, triage_severity, news2_score,
                        qsofa_score, curb65_score, gcs_total, body_system,
                        red_flags, escalation_required, top_icd_code,
                        top_condition_name, patient_report_length,
                        safety_compliant, llm_provider_used, session_data_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        session_id,
                        datetime.now(timezone.utc).isoformat(),
                        triage_severity,
                        news2_score,
                        qsofa_score,
                        curb65_score,
                        gcs_total,
                        body_system,
                        json.dumps(red_flags),
                        int(escalation_required),
                        top_icd_code,
                        top_condition_name,
                        patient_report_length,
                        int(safety_compliant),
                        llm_provider_used,
                        json.dumps(session_data),
                    ),
                )

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM diagnosis_sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if row:
                    d = dict(row)
                    d["red_flags"] = json.loads(d.get("red_flags") or "[]")
                    d["session_data"] = json.loads(d.get("session_data_json") or "{}")
                    return d
        return None

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM diagnosis_sessions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        use_case: str = "",
    ) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO llm_cost_log (logged_at, provider, model, prompt_tokens, completion_tokens, cost_usd, use_case) VALUES (?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), provider, model, prompt_tokens, completion_tokens, cost_usd, use_case),
                )

    def get_cost_summary(self, days: int = 30) -> dict:
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT provider, SUM(cost_usd) as total, COUNT(*) as calls,
                       SUM(prompt_tokens+completion_tokens) as total_tokens
                       FROM llm_cost_log
                       WHERE logged_at > datetime('now', ?)
                       GROUP BY provider""",
                    (f"-{days} days",),
                ).fetchall()
                return {row["provider"]: dict(row) for row in rows}

    def is_known_paper(self, identifier: str) -> bool:
        h = hashlib.sha256(identifier.encode()).hexdigest()
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT hash FROM knowledge_hashes WHERE hash=?", (h,)
                ).fetchone()
                return row is not None

    def mark_paper_known(self, identifier: str, source_url: str = "", title: str = "") -> None:
        h = hashlib.sha256(identifier.encode()).hexdigest()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_hashes (hash, source_url, title, added_at) VALUES (?,?,?,?)",
                    (h, source_url, title, datetime.now(timezone.utc).isoformat()),
                )

    def get_stats(self) -> dict:
        with self._lock:
            with self._get_conn() as conn:
                total = conn.execute("SELECT COUNT(*) FROM diagnosis_sessions").fetchone()[0]
                by_severity = conn.execute(
                    "SELECT triage_severity, COUNT(*) as cnt FROM diagnosis_sessions GROUP BY triage_severity"
                ).fetchall()
                escalations = conn.execute(
                    "SELECT COUNT(*) FROM diagnosis_sessions WHERE escalation_required=1"
                ).fetchone()[0]
                papers = conn.execute("SELECT COUNT(*) FROM knowledge_hashes").fetchone()[0]
                return {
                    "total_sessions": total,
                    "escalations": escalations,
                    "by_severity": {row["triage_severity"]: row["cnt"] for row in by_severity},
                    "known_papers": papers,
                }
