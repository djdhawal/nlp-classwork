"""Appointments in SQLite. The agent's four tools call these methods."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DATE_FMT = "%Y-%m-%d %H:%M"
FIELDS = ("date", "location", "title", "description")


def normalize_date(value: str) -> str:
    """Accept 'YYYY-MM-DD HH:MM' (also with a T or seconds) and return 'YYYY-MM-DD HH:MM'."""
    text = str(value).strip().replace("T", " ")
    for fmt in (DATE_FMT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime(DATE_FMT)
        except ValueError:
            pass
    raise ValueError(f"date must look like YYYY-MM-DD HH:MM, got {value!r}")


class AppointmentStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the agent thread and the web page, guarded by a lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS appointments (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       date TEXT NOT NULL,
                       location TEXT NOT NULL,
                       title TEXT NOT NULL,
                       description TEXT DEFAULT '',
                       created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
            )

    def create(self, date: str, location: str, title: str, description: str = "") -> dict:
        if not str(location or "").strip():
            raise ValueError("location is required")
        if not str(title or "").strip():
            raise ValueError("title is required")
        date = normalize_date(date)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO appointments (date, location, title, description) VALUES (?, ?, ?, ?)",
                (date, location.strip(), title.strip(), (description or "").strip()),
            )
            new_id = cur.lastrowid
        return self.get(new_id)

    def get(self, appointment_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, date, location, title, description FROM appointments WHERE id = ?",
                (int(appointment_id),),
            ).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, date, location, title, description FROM appointments ORDER BY date, id"
            ).fetchall()
        return [dict(r) for r in rows]

    def update(self, appointment_id: int, **changes) -> dict:
        changes = {k: v for k, v in changes.items() if k in FIELDS and v is not None}
        if not changes:
            raise ValueError("no fields to change")
        if "date" in changes:
            changes["date"] = normalize_date(changes["date"])
        if self.get(appointment_id) is None:
            raise KeyError(f"no appointment with id {appointment_id}")
        sets = ", ".join(f"{k} = ?" for k in changes)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE appointments SET {sets} WHERE id = ?",
                               (*changes.values(), int(appointment_id)))
        return self.get(appointment_id)

    def delete(self, appointment_id: int) -> dict:
        appt = self.get(appointment_id)
        if appt is None:
            raise KeyError(f"no appointment with id {appointment_id}")
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM appointments WHERE id = ?", (int(appointment_id),))
        return appt

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM appointments")
