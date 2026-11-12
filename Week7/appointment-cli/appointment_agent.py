"""Appointment assistant: interactive CLI agent on a local Ollama model (settings from ../config.yaml).

Custom tools keep appointments in a local SQLite file (tools.appointments_db in config.yaml).
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from strands import Agent, tool
from strands.models.ollama import OllamaModel

logging.getLogger("strands").setLevel(logging.WARNING)

cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
DB_PATH = cfg["tools"]["appointments_db"]
model = OllamaModel(
    host=cfg["ollama"]["url"],
    model_id=cfg["ollama"]["models"][0],
    temperature=cfg["strands"]["temperature"],
    options={"num_ctx": cfg["strands"]["num_ctx"]},
)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS appointments (
        id TEXT PRIMARY KEY, date TEXT, location TEXT, title TEXT, description TEXT)""")
    return conn


def _check_date(date: str) -> None:
    try:
        datetime.strptime(date, "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("Date must be in format 'YYYY-MM-DD HH:MM'")


@tool
def current_time() -> str:
    """Get the current local date and time in 'YYYY-MM-DD HH:MM' format, plus the weekday."""
    return datetime.now().strftime("%Y-%m-%d %H:%M (%A)")


@tool
def create_appointment(date: str, location: str, title: str, description: str) -> str:
    """Create a new personal appointment.

    Args:
        date: Date and time of the appointment (format: YYYY-MM-DD HH:MM).
        location: Location of the appointment.
        title: Title of the appointment.
        description: Description of the appointment.

    Returns:
        The ID of the newly created appointment.
    """
    _check_date(date)
    appointment_id = str(uuid.uuid4())[:8]
    with _db() as conn:
        conn.execute("INSERT INTO appointments VALUES (?, ?, ?, ?, ?)",
                     (appointment_id, date, location, title, description))
    return f"Appointment with id {appointment_id} created for {date} at {location}."


@tool
def list_appointments() -> str:
    """List all appointments, ordered by date, as a JSON list."""
    with _db() as conn:
        rows = conn.execute("SELECT * FROM appointments ORDER BY date").fetchall()
    return json.dumps([dict(r) for r in rows]) if rows else "No appointment available"


@tool
def update_appointment(appointment_id: str, date: str = "", location: str = "",
                       title: str = "", description: str = "") -> str:
    """Update an appointment by id. Only the fields you pass are changed.

    Args:
        appointment_id: The appointment id.
        date: New date and time (format: YYYY-MM-DD HH:MM).
        location: New location.
        title: New title.
        description: New description.
    """
    fields = {k: v for k, v in dict(date=date, location=location, title=title, description=description).items() if v}
    if not fields:
        raise ValueError("Nothing to update: pass at least one field.")
    if "date" in fields:
        _check_date(fields["date"])
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _db() as conn:
        cur = conn.execute(f"UPDATE appointments SET {sets} WHERE id = ?", [*fields.values(), appointment_id])
    if cur.rowcount == 0:
        raise ValueError(f"Appointment {appointment_id} not found.")
    return f"Appointment {appointment_id} updated: {fields}"


agent = Agent(
    model=model,
    system_prompt="""You are a helpful personal assistant that manages my appointments and calendar.
Rules:
- Dates passed to tools must look like 'YYYY-MM-DD HH:MM'. Call current_time first when you need today's date.
- To change an existing appointment, call update_appointment with its id. Never create a duplicate.
- Always tell me the appointment id so I can update it later.""",
    tools=[current_time, create_appointment, list_appointments, update_appointment],
)

if __name__ == "__main__":
    print(f"\nAppointment assistant (database: {DB_PATH}). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("\nYou > ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        print("\nAssistant > ", end="")
        agent(user_input)   # the reply streams to the terminal as it is generated
        print()
    print("Bye!")
