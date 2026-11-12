"""Long-term memory: personal facts, searchable by meaning (Ollama embeddings + Qdrant)."""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import resolve_path

log = logging.getLogger("va.memory")

EXTRACT_PROMPT = """You pull long-term personal facts out of one message a user said to their voice assistant.
Keep only lasting facts about the user: preferences, habits, people in their life, places they go, things they like or dislike.
Do NOT keep appointments, bookings, dates, times of specific events, questions, or commands to the assistant.
Write each fact as a short sentence about "the user", e.g. "The user prefers morning meetings."
Return JSON: {"facts": ["..."]}. Return {"facts": []} if there is nothing worth keeping.

Message: """

# Facts that look like scheduling go to the appointments database, never to memory.
_SCHEDULE = re.compile(
    r"\b(appointment|booked?|booking|schedul\w*|calendar|reschedul\w*|cancel\w*|tomorrow|today|tonight|"
    r"next (week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|\d{1,2}(:\d\d)?\s*(am|pm|a\.m\.|p\.m\.)|"
    r"\d{4}-\d{2}-\d{2})\b",
    re.I,
)
# Explicit "remember that ..." requests are only refused if they are clearly a booking.
_BOOKING = re.compile(r"\b(appointment|booked?|booking|calendar|\d{4}-\d{2}-\d{2})\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'd|my|me|mine|remember)\b", re.I)


def looks_like_schedule(text: str, strict: bool = True) -> bool:
    return bool((_SCHEDULE if strict else _BOOKING).search(text))


class MemoryStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m, e = cfg["memory"], cfg["embeddings"]
        self.collection = m["collection"]
        self.top_k, self.min_score, self.dup_score = m["top_k"], m["min_score"], m["duplicate_score"]
        self.embed_model = e["model"]
        self.q_prefix, self.d_prefix = e.get("query_prefix", ""), e.get("document_prefix", "")
        self.llm = ollama.Client(host=cfg["llm"]["host"])
        if m.get("qdrant_path"):
            path = resolve_path(cfg, m["qdrant_path"])
            path.mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=str(path))
        else:
            self.qdrant = QdrantClient(url=m["qdrant_url"])
        self._lock = threading.Lock()
        self._ready = False

    # -- vectors ------------------------------------------------------------
    def embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        resp = self.llm.embed(model=self.embed_model, input=[prefix + t for t in texts],
                              keep_alive=self.cfg["llm"].get("keep_alive"))
        return resp.embeddings

    def _ensure_collection(self, dim: int) -> None:
        if self._ready:
            return
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self._ready = True

    # -- public API ---------------------------------------------------------
    def add(self, text: str, source: str = "user") -> dict:
        """Store one fact. Returns {"stored": bool, ...}; near-duplicates are skipped."""
        text = text.strip()
        if not text:
            return {"stored": False, "reason": "empty"}
        if looks_like_schedule(text, strict=(source != "user")):
            return {"stored": False, "reason": "appointments belong in the calendar, not memory"}
        vec = self.embed([text], self.d_prefix)[0]
        with self._lock:
            self._ensure_collection(len(vec))
            hits = self._query(vec, 1)
            if hits and hits[0]["score"] >= self.dup_score:
                return {"stored": False, "reason": "already known", "existing": hits[0]["text"]}
            pid = str(uuid.uuid4())
            self.qdrant.upsert(self.collection, points=[PointStruct(
                id=pid, vector=vec,
                payload={"text": text, "source": source, "created_at": datetime.now().isoformat(timespec="seconds")})])
        log.info("memory stored (%s): %s", source, text)
        return {"stored": True, "id": pid, "text": text}

    def _query(self, vec, k: int) -> list[dict]:
        res = self.qdrant.query_points(self.collection, query=vec, limit=k, with_payload=True)
        return [{"id": str(p.id), "text": p.payload["text"], "score": float(p.score)} for p in res.points]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        if not query.strip() or not self.qdrant.collection_exists(self.collection):
            return []
        vec = self.embed([query], self.q_prefix)[0]
        with self._lock:
            self._ensure_collection(len(vec))
            hits = self._query(vec, k or self.top_k)
        return [h for h in hits if h["score"] >= self.min_score]

    def all(self) -> list[dict]:
        if not self.qdrant.collection_exists(self.collection):
            return []
        points, _ = self.qdrant.scroll(self.collection, limit=500, with_payload=True)
        rows = [{"id": str(p.id), **p.payload} for p in points]
        return sorted(rows, key=lambda r: r.get("created_at", ""))

    def delete(self, point_id: str) -> None:
        self.qdrant.delete(self.collection, points_selector=[point_id])

    def clear(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)
        self._ready = False

    # -- background fact extraction -----------------------------------------
    def extract_facts(self, user_text: str, model: str) -> list[str]:
        """Ask the chat model for lasting personal facts in what the user said."""
        if not _FIRST_PERSON.search(user_text):
            return []          # "yes", "what is on my calendar"... -> skip the LLM call entirely
        kwargs = {}
        if self.cfg["llm"].get("think") is not None:
            kwargs["think"] = self.cfg["llm"]["think"]
        resp = self.llm.chat(model=model, messages=[{"role": "user", "content": EXTRACT_PROMPT + user_text}],
                             format="json", options={"temperature": 0}, keep_alive=self.cfg["llm"].get("keep_alive"),
                             **kwargs)
        try:
            facts = json.loads(resp.message.content).get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            log.warning("fact extraction returned non-JSON: %r", resp.message.content[:200])
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip() and not looks_like_schedule(f)]

    def extract_and_store(self, user_text: str, model: str) -> list[dict]:
        try:
            return [self.add(f, source="extracted") for f in self.extract_facts(user_text, model)]
        except Exception:
            log.exception("background memory extraction failed")
            return []
