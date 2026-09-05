"""Append-only per-request telemetry; unknown usage never becomes zero cost."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Pricing


def redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(r"(?i)(Bearer\s+)\S+", r"\1[REDACTED]", text)


def sanitize(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def cost_usd(prompt: int | None, completion: int | None, cached: int, pricing: Pricing | None) -> float | None:
    if prompt is None or completion is None or pricing is None:
        return None
    cached = min(max(cached, 0), prompt)
    return ((prompt - cached) * pricing.input_per_million + cached * pricing.cached_input_per_million
            + completion * pricing.output_per_million) / 1_000_000


class Telemetry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, timestamp TEXT, "
                       "run_id TEXT, kind TEXT, payload TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS events_run ON events(run_id)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        return db

    def record(self, run_id: str, kind: str, **payload) -> dict:
        event = {"id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "run_id": run_id, "kind": kind, **payload}
        safe = json.loads(json.dumps(sanitize(event), default=str))
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                       (safe["id"], safe["timestamp"], run_id, kind, json.dumps(safe)))
        return safe

    def recent(self, limit: int = 500) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]
