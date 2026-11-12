"""The always-on assistant: mic -> STT -> wake word -> agent -> TTS, plus state for the web page.

One Assistant owns three background threads, started once:
  loader   loads the models, then exits
  listener reads the microphone and turns utterances into commands
  worker   runs commands one at a time (spoken or typed)
The web page only reads `snapshot()` and calls submit / set_muted / stop_speaking.
"""
from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from datetime import datetime

from .agent import Agent
from .appointments import AppointmentStore
from .config import resolve_path, setup_logging
from .speech_out import Speaker
from .wake import WakeWord

log = logging.getLogger("va.assistant")

# Whisper often "hears" these in background noise.
_HALLUCINATIONS = re.compile(r"^\W*(thank you|thanks for watching|you|bye|okay|\.+)\W*$", re.I)


class Assistant:
    def __init__(self, cfg: dict, voice: bool = True, speak: bool = True, model: str | None = None,
                 stt: bool | None = None):
        """voice: listen on the microphone. stt: load speech to text even without a microphone
        (Colab: clips recorded in the browser are passed to handle_audio)."""
        self.cfg, self.voice, self.speak, self.model_name = cfg, voice, speak, model
        self.stt = voice if stt is None else stt
        setup_logging(cfg)
        self.store = AppointmentStore(resolve_path(cfg, cfg["appointments"]["db_path"]))
        self.memory = None
        self.agent: Agent | None = None
        self.speaker = Speaker(cfg, enabled=speak)
        self.wake = WakeWord(cfg)
        self.mic = self.transcriber = None

        self._lock = threading.RLock()
        self._jobs: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self.turns: list[dict] = []
        self.heard: list[dict] = []        # every transcript, including ignored ones
        self.loading, self.busy, self.muted = True, False, False
        self.error: str | None = None
        self.notes: list[str] = []
        self.follow_up_until = 0.0
        self.noise_threshold: float | None = None
        self._started = False

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "Assistant":
        if self._started:
            return self
        self._started = True
        threading.Thread(target=self._load, name="loader", daemon=True).start()
        return self

    def _load(self) -> None:
        try:
            try:
                from .memory import MemoryStore
                self.memory = MemoryStore(self.cfg)
                self.memory.all()                       # fails fast if Qdrant is not running
            except Exception as exc:
                self.memory = None
                self.notes.append(f"Memory is off: {exc}")
                log.warning("memory disabled: %s", exc)
            self.agent = Agent(self.cfg, self.store, self.memory, model=self.model_name)
            self.agent.warm_up()
            if self.stt or self.voice:
                from .speech_in import Transcriber
                self.transcriber = Transcriber(self.cfg)
            if self.voice:
                from .speech_in import Microphone
                self.mic = Microphone(self.cfg).start()
                threading.Thread(target=self._listen, name="listener", daemon=True).start()
            threading.Thread(target=self._work, name="worker", daemon=True).start()
            log.info("assistant ready (model=%s, voice=%s)", self.agent.model, self.voice)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("start-up failed")
        finally:
            self.loading = False

    def wait_ready(self, timeout: float = 300) -> None:
        end = time.monotonic() + timeout
        while self.loading and time.monotonic() < end:
            time.sleep(0.1)
        if self.error:
            raise RuntimeError(self.error)

    # -- status -----------------------------------------------------------------
    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.loading:
            return "loading"
        if self.speaker.speaking:
            return "speaking"
        if self.busy or not self._jobs.empty():
            return "thinking"
        if self.muted:
            return "muted"
        return "listening" if self.voice else "ready"

    def _mic_closed(self) -> bool:
        delay = self.cfg["audio"]["unmute_delay_seconds"]
        return (self.muted or self.busy or self.speaker.speaking
                or time.monotonic() - self.speaker.last_done < delay)

    # -- controls -----------------------------------------------------------------
    def submit(self, text: str, source: str = "typed", heard: str | None = None) -> int:
        turn_id = next(self._ids)
        with self._lock:
            self.turns.append({"id": turn_id, "at": datetime.now().strftime("%H:%M:%S"), "source": source,
                               "heard": heard if heard is not None else text, "command": text,
                               "tools": [], "reply": "", "state": "queued", "seconds": None, "first_audio": None})
        self._jobs.put(turn_id)
        return turn_id

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            self.speaker.stop()

    def stop_speaking(self) -> None:
        self.speaker.stop()

    def run_text(self, text: str, timeout: float = 300) -> dict:
        """Submit a typed command and wait for its turn to finish (used by the CLI and the checks)."""
        tid = self.submit(text)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            turn = self.turn(tid)
            if turn["state"] in ("done", "error"):
                return turn
            time.sleep(0.05)
        raise TimeoutError(text)

    def turn(self, turn_id: int) -> dict:
        with self._lock:
            return next(t for t in self.turns if t["id"] == turn_id)

    def snapshot(self) -> dict:
        with self._lock:
            turns = [dict(t, tools=list(t["tools"])) for t in self.turns[-30:]]
            heard = list(self.heard[-20:])
        return {"status": self.status, "turns": turns, "heard": heard, "error": self.error, "notes": list(self.notes),
                "follow_up": max(0.0, self.follow_up_until - time.monotonic()),
                "model": self.agent.model if self.agent else self.cfg["llm"]["model"],
                "threshold": self.noise_threshold}

    # -- worker: one command at a time ---------------------------------------------------
    def _work(self) -> None:
        while True:
            turn_id = self._jobs.get()
            turn = self.turn(turn_id)
            self.busy = True
            turn["state"] = "thinking"
            self.speaker.reset_timer()
            t0 = time.perf_counter()
            try:
                result = self.agent.run(turn["command"], on_text=self.speaker.feed,
                                        on_tool=lambda ev: turn["tools"].append(ev))
                self.speaker.flush()
                turn.update(reply=result.reply, seconds=round(result.seconds, 2), state="done",
                            memories=[m["text"] for m in result.memories])
            except Exception as exc:
                log.exception("turn failed")
                turn.update(reply=f"Error: {exc}", state="error")
                self.speaker.say("Sorry, something went wrong.")
            finally:
                self.busy = False
            self.speaker.wait()
            if self.speaker.first_audio_at:
                turn["first_audio"] = round(self.speaker.first_audio_at - t0, 2)
            self.follow_up_until = time.monotonic() + self.cfg["assistant"]["follow_up_seconds"]
            if self.voice and not self.muted and self._jobs.empty() and self.cfg["audio"].get("start_tone"):
                from .speech_in import beep
                beep(self.cfg)
            # Memory extraction after speaking, in the background, so it never slows a reply.
            if self.memory is not None and self.cfg["memory"].get("extract_in_background") and turn["state"] == "done":
                threading.Thread(target=self._extract, args=(turn,), name="memory", daemon=True).start()

    def _extract(self, turn: dict) -> None:
        stored = [r for r in self.memory.extract_and_store(turn["command"], self.agent.model) if r.get("stored")]
        if stored:
            turn["extracted"] = [r["text"] for r in stored]

    # -- listener: mic -> utterance -> text -> wake word -----------------------------------------
    def _listen(self) -> None:
        from .speech_in import Segmenter
        seg = Segmenter(self.cfg)
        was_closed = False
        while True:
            frame = self.mic.read()
            if frame is None:
                continue
            if self._mic_closed():           # never hear ourselves: drop audio while speaking/thinking/muted
                if not was_closed:
                    seg.reset()
                was_closed = True
                self.mic.drain()
                continue
            if was_closed:
                seg.reset()
                was_closed = False
            audio = seg.feed(frame)
            self.noise_threshold = seg.threshold if seg.calibrated else None
            if audio is None:
                continue
            try:
                self._handle_utterance(self.transcriber.transcribe(audio))
            except Exception:
                log.exception("transcription failed")

    def handle_audio(self, audio, started_at: float | None = None) -> str:
        """Transcribe one utterance recorded elsewhere and act on it like the listener would.

        started_at (time.monotonic()) is when the speech started, for the follow-up window."""
        text = self.transcriber.transcribe(audio)
        self._handle_utterance(text, started_at)
        return text

    def _handle_utterance(self, text: str, started_at: float | None = None) -> None:
        if not text or _HALLUCINATIONS.match(text):
            return
        command = self.wake.find(text)
        in_follow_up = (started_at or time.monotonic()) < self.follow_up_until
        if command is None and in_follow_up:
            command = text
        action = "ignored (no wake word)" if command is None else "wake only" if command == "" else "command"
        with self._lock:
            self.heard.append({"at": datetime.now().strftime("%H:%M:%S"), "text": text, "action": action})
        log.info("heard %r -> %s", text, action)
        if command is None:
            return
        if command == "":
            self.speaker.say("Yes?")
            self.speaker.wait()
            self.follow_up_until = time.monotonic() + self.cfg["assistant"]["follow_up_seconds"]
            return
        self.follow_up_until = 0.0
        self.submit(command, source="voice", heard=text)
