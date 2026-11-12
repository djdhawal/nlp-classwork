"""The agent: an Ollama chat model in a tool-calling loop, with appointments and memory.

Works with any Ollama model that supports tools; the model name comes from the config.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import ollama

from .appointments import AppointmentStore

log = logging.getLogger("va.agent")

# ---------------------------------------------------------------------------
# Tool schemas. Names and arguments are fixed by the exercise.
# ---------------------------------------------------------------------------
_DATE = {"type": "string", "description": "Date and time as YYYY-MM-DD HH:MM, 24-hour clock"}
TOOLS = [
    {"type": "function", "function": {
        "name": "create_appointment",
        "description": "Save one new appointment. Only call this when the user said the date, time AND location.",
        "parameters": {"type": "object", "required": ["date", "location", "title"], "properties": {
            "date": _DATE,
            "location": {"type": "string", "description": "Where it is, exactly as the user said it"},
            "title": {"type": "string", "description": "Short title, e.g. 'Dentist'"},
            "description": {"type": "string", "description": "Optional extra details"}}}}},
    {"type": "function", "function": {
        "name": "list_appointments",
        "description": "Return all appointments ordered by date, with their ids.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "update_appointment",
        "description": "Change an existing appointment. Pass only the fields that change.",
        "parameters": {"type": "object", "required": ["appointment_id"], "properties": {
            "appointment_id": {"type": "integer", "description": "Id from list_appointments"},
            "date": _DATE,
            "location": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "delete_appointment",
        "description": "Delete one appointment. Only call after the user answered yes to your confirmation question.",
        "parameters": {"type": "object", "required": ["appointment_id"], "properties": {
            "appointment_id": {"type": "integer", "description": "Id from list_appointments"}}}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Store a lasting personal fact or preference about the user, e.g. 'The user prefers morning meetings'. Never for appointments.",
        "parameters": {"type": "object", "required": ["fact"], "properties": {
            "fact": {"type": "string", "description": "One short sentence about the user"}}}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM_PROMPT = """You are {name}, a voice assistant that manages the user's appointments. Everything you write is read aloud.

Current date and time: {now}.
Dates for the next week:
{calendar}

Rules:
1. Tool dates are YYYY-MM-DD HH:MM on a 24-hour clock: 3 pm is 15:00, noon is 12:00. A weekday name means the next date with that name in the list above.
2. To book you need a date, a time, a title and a location. If the user did not say where, ask one short question about the location and do not call create_appointment. Never invent or guess a location.
3. Never say an appointment was booked, moved or deleted unless the tool you just called returned "ok": true. When you book, say the id from the tool result.
4. To move or change an appointment, call list_appointments to find its id, then call update_appointment with only the fields that change. Never create a new appointment to move one.
5. To delete: call list_appointments, then ask "Delete the <title> on <day> at <time>?" and stop. Call delete_appointment only after the user says yes. If they say no, keep it.
6. For questions about the calendar, call list_appointments and answer from its result.
7. When the user asks you to remember something about themselves, call remember. Appointments never go into memory.
8. Answer in one or two short sentences of plain text. No lists, no markdown, no emojis. Say times the way people speak, like 3 PM.
{memories}"""

_AFFIRM = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|confirm(ed)?|correct|go ahead|do it|please do|delete it|that's right|right)\b", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class TurnResult:
    user: str
    reply: str = ""
    tool_events: list[dict] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    first_text_seconds: float | None = None
    model: str = ""


class _TextGate:
    """Streams text deltas out, unless the reply starts like a JSON tool call or a <think> block."""

    def __init__(self, sink: Callable[[str], None] | None):
        self.sink, self.mode, self.buf = sink, None, ""

    def feed(self, delta: str) -> None:
        if not delta:
            return
        self.buf += delta
        if self.mode is None:
            stripped = self.buf.lstrip()
            if not stripped:
                return
            self.mode = "hold" if stripped[0] in "{[`<" else "stream"
            if self.mode == "stream" and self.sink:
                self.sink(self.buf)
            return
        if self.mode == "stream" and self.sink:
            self.sink(delta)


def _parse_text_tool_call(content: str) -> list[dict]:
    """Small models sometimes write the tool call as JSON text instead of a real tool call."""
    text = content.strip().strip("`")
    text = re.sub(r"^json\s*", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    objs = obj if isinstance(obj, list) else [obj]
    calls = []
    for o in objs:
        if isinstance(o, dict) and o.get("name") in TOOL_NAMES:
            args = o.get("arguments", o.get("parameters", {})) or {}
            calls.append({"name": o["name"], "arguments": args})
    return calls


class Agent:
    def __init__(self, cfg: dict, store: AppointmentStore, memory=None, model: str | None = None,
                 client=None, clock: Callable[[], datetime] | None = None):
        self.cfg, self.store, self.memory = cfg, store, memory
        self.model = model or cfg["llm"]["model"]
        self.client = client or ollama.Client(host=cfg["llm"]["host"])
        tz = cfg["assistant"].get("timezone")
        self.tz = ZoneInfo(tz) if tz else None
        self.clock = clock or (lambda: datetime.now(self.tz).replace(tzinfo=None))
        self.history: list[dict] = []
        self.guards = bool(cfg["llm"].get("guards", True))

    # -- prompt ---------------------------------------------------------------
    def system_prompt(self, memories: list[dict]) -> str:
        now = self.clock()
        days = []
        for i in range(8):
            d = now + timedelta(days=i)
            label = "today" if i == 0 else "tomorrow" if i == 1 else d.strftime("%A")
            days.append(f"- {label}: {d:%A %Y-%m-%d}")
        mem = ""
        if memories:
            mem = "\nWhat you know about the user (from memory):\n" + "\n".join(f"- {m['text']}" for m in memories)
        return SYSTEM_PROMPT.format(name=self.cfg["assistant"]["name"], now=f"{now:%A, %Y-%m-%d %H:%M}",
                                    calendar="\n".join(days), memories=mem)

    def warm_up(self) -> None:
        """Load the model into memory now, so the first command is fast."""
        self.client.chat(model=self.model, messages=[{"role": "user", "content": "hi"}],
                         options={"num_predict": 1}, keep_alive=self.cfg["llm"].get("keep_alive"))

    def reset(self) -> None:
        self.history.clear()

    # -- tools -----------------------------------------------------------------
    def _recent_user_text(self, current: str, memories: list[dict]) -> str:
        users = [m["content"] for m in self.history if m["role"] == "user"][-3:]
        return " ".join(users + [current] + [m["text"] for m in memories]).lower()

    def _last_assistant_text(self) -> str:
        for m in reversed(self.history):
            if m["role"] == "assistant" and m.get("content"):
                return m["content"]
        return ""

    def call_tool(self, name: str, args: dict, ctx: dict) -> dict:
        """Run one tool and return a JSON-able result with "ok"."""
        try:
            if name == "create_appointment":
                location = str(args.get("location") or "").strip()
                if not location:
                    return {"ok": False, "error": "No location given. Ask the user where it is."}
                if self.guards and not _grounded(location, ctx["user_text"]):
                    return {"ok": False, "error": f"The user never said the location {location!r}. Ask the user where it is."}
                if self.guards:   # the same booking twice in one conversation -> return the first one
                    for a in self.store.list():
                        if (a["date"], a["title"].lower(), a["location"].lower()) == (
                                _safe_date(args.get("date")), str(args.get("title", "")).strip().lower(), location.lower()):
                            return {"ok": True, "id": a["id"], "appointment": a, "note": "already booked"}
                appt = self.store.create(args.get("date", ""), location, args.get("title", ""), args.get("description") or "")
                return {"ok": True, "id": appt["id"], "appointment": appt}
            if name == "list_appointments":
                appts = self.store.list()
                return {"ok": True, "count": len(appts), "appointments": appts}
            if name == "update_appointment":
                aid = int(args.get("appointment_id"))
                changes = {k: args[k] for k in ("date", "location", "title", "description") if args.get(k)}
                return {"ok": True, "appointment": self.store.update(aid, **changes)}
            if name == "delete_appointment":
                aid = int(args.get("appointment_id"))
                appt = self.store.get(aid)
                if appt is None:
                    return {"ok": False, "error": f"No appointment with id {aid}."}
                if self.guards and not (_AFFIRM.search(ctx["user_text_now"]) and "?" in self._last_assistant_text()):
                    return {"ok": False, "error": "Not deleted yet. First ask the user to confirm, e.g. "
                            f"'Delete the {appt['title']} on {appt['date']}?', and wait for yes.", "appointment": appt}
                return {"ok": True, "deleted": self.store.delete(aid)}
            if name == "remember":
                if self.memory is None:
                    return {"ok": False, "error": "Memory is not available."}
                res = self.memory.add(str(args.get("fact", "")), source="user")
                return {"ok": res.get("stored", False) or res.get("reason") == "already known", **res}
            return {"ok": False, "error": f"Unknown tool {name}."}
        except (ValueError, KeyError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}

    # -- the loop ----------------------------------------------------------------
    def run(self, user_text: str, on_text: Callable[[str], None] | None = None,
            on_tool: Callable[[dict], None] | None = None) -> TurnResult:
        """One user turn. `on_text` gets reply text as it streams; `on_tool` gets each tool call and result."""
        t0 = time.perf_counter()
        result = TurnResult(user=user_text, model=self.model)
        memories = []
        if self.memory is not None:
            try:
                memories = self.memory.search(user_text)
            except Exception:
                log.exception("memory search failed")
        result.memories = memories
        ctx = {"user_text": self._recent_user_text(user_text, memories), "user_text_now": user_text}

        def text_sink(delta: str) -> None:
            if result.first_text_seconds is None:
                result.first_text_seconds = time.perf_counter() - t0
            if on_text:
                on_text(delta)

        turn = [{"role": "user", "content": user_text}]
        kwargs = {"options": {"temperature": self.cfg["llm"].get("temperature", 0)},
                  "keep_alive": self.cfg["llm"].get("keep_alive")}
        if self.cfg["llm"].get("think") is not None:
            kwargs["think"] = self.cfg["llm"]["think"]

        reply = ""
        for _ in range(int(self.cfg["llm"].get("max_tool_rounds", 6))):
            messages = [{"role": "system", "content": self.system_prompt(memories)}] + self.history + turn
            gate, content, calls = _TextGate(text_sink), "", []
            for chunk in self.client.chat(model=self.model, messages=messages, tools=TOOLS, stream=True, **kwargs):
                msg = chunk.message
                if msg.content:
                    content += msg.content
                    gate.feed(msg.content)
                for tc in msg.tool_calls or []:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    calls.append({"name": tc.function.name, "arguments": dict(args or {})})
            content = _THINK.sub("", content).strip()
            if not calls and gate.mode == "hold":
                calls = _parse_text_tool_call(content)
                if calls:
                    content = ""
                elif on_text and content:   # held text that was not a tool call after all
                    text_sink(content)
            if not calls:
                reply = content
                break
            turn.append({"role": "assistant", "content": content,
                         "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for c in calls:
                out = self.call_tool(c["name"], c["arguments"], ctx)
                event = {"tool": c["name"], "arguments": c["arguments"], "result": out}
                result.tool_events.append(event)
                log.info("tool %s(%s) -> %s", c["name"], json.dumps(c["arguments"]), json.dumps(out)[:300])
                if on_tool:
                    on_tool(event)
                turn.append({"role": "tool", "tool_name": c["name"], "content": json.dumps(out)})
        else:
            reply = "Sorry, I could not finish that."
            text_sink(reply)

        if not reply:
            reply = "Sorry, I did not get that."
            text_sink(reply)
        turn.append({"role": "assistant", "content": reply})
        self.history.extend(turn)
        self._trim_history()
        result.reply = reply
        result.seconds = time.perf_counter() - t0
        log.info("turn model=%s %.2fs user=%r reply=%r", self.model, result.seconds, user_text, reply)
        return result

    def _trim_history(self) -> None:
        keep = int(self.cfg["assistant"].get("history_turns", 8))
        starts = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(starts) > keep:
            self.history = self.history[starts[-keep]:]


def _safe_date(value) -> str:
    from .appointments import normalize_date
    try:
        return normalize_date(value)
    except ValueError:
        return ""


_STOP = {"the", "a", "an", "at", "in", "on", "of", "to", "my", "street", "st", "ave", "avenue", "road", "rd"}


def _grounded(location: str, said: str) -> bool:
    """True if some word of the location was actually said by the user (or is in memory)."""
    words = [w for w in re.findall(r"[a-z0-9]+", location.lower()) if w not in _STOP]
    if not words:
        return True
    said_words = set(re.findall(r"[a-z0-9]+", said))
    return any(w in said_words for w in words)
