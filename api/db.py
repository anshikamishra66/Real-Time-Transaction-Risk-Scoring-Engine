"""SQLite-backed transaction log: powers the live feed and the review queue.

SQLite (not in-memory) so the review queue survives an API restart during a
demo, while staying a single file with zero setup -- appropriate for this
system's scale and deployment shape, not a claim that it would scale to a
production payment processor's transaction volume.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    geo_region TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL,
    risk_score INTEGER NOT NULL,
    risk_tier TEXT NOT NULL,
    action TEXT NOT NULL,
    supervised_prob REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    rules_triggered TEXT NOT NULL,
    features TEXT NOT NULL,
    shap_explanation TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_created_at ON transactions (created_at);
CREATE INDEX IF NOT EXISTS idx_tx_risk_tier ON transactions (risk_tier);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def reset_db() -> None:
    with get_conn() as conn:
        conn.executescript("DELETE FROM transactions;")


def insert_transaction(record: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO transactions (
                transaction_id, account_id, device_id, geo_region, amount, timestamp,
                risk_score, risk_tier, action, supervised_prob, anomaly_score,
                rules_triggered, features, shap_explanation, review_status, created_at
            ) VALUES (:transaction_id, :account_id, :device_id, :geo_region, :amount, :timestamp,
                :risk_score, :risk_tier, :action, :supervised_prob, :anomaly_score,
                :rules_triggered, :features, :shap_explanation, :review_status, :created_at)
            """,
            {
                **record,
                "rules_triggered": json.dumps(record["rules_triggered"]),
                "features": json.dumps(record["features"]),
                "shap_explanation": json.dumps(record["shap_explanation"]) if record.get("shap_explanation") else None,
                "review_status": record.get("review_status", "pending"),
                "created_at": record.get("created_at", datetime.now(timezone.utc).isoformat()),
            },
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["rules_triggered"] = json.loads(d["rules_triggered"])
    d["features"] = json.loads(d["features"])
    d["shap_explanation"] = json.loads(d["shap_explanation"]) if d.get("shap_explanation") else None
    return d


def fetch_recent(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def fetch_review_queue(status: Optional[str] = "pending", limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                """SELECT * FROM transactions
                   WHERE risk_tier IN ('medium', 'high') AND review_status = ?
                   ORDER BY risk_score DESC, created_at DESC LIMIT ?""",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM transactions WHERE risk_tier IN ('medium', 'high')
                   ORDER BY risk_score DESC, created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_review_status(transaction_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE transactions SET review_status = ? WHERE transaction_id = ?",
            (status, transaction_id),
        )
