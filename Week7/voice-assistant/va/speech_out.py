"""Text to speech: split streamed text into sentences, speak each one as soon as it is complete.

Every backend runs as a child process, so stop() can kill it at once.
"""
from __future__ import annotations

import logging
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

log = logging.getLogger("va.tts")

# ---------------------------------------------------------------------------
# Text clean-up and sentence splitting
# ---------------------------------------------------------------------------
_ABBREV = {"dr", "mr", "mrs", "ms", "st", "ave", "jr", "sr", "vs", "etc", "e.g", "i.e", "a.m", "p.m", "no"}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


def clean_for_speech(text: str) -> str:
    """Remove things a voice would read badly: markdown, code ticks, bullets, links, emojis."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*`_~]+", "", text)
    text = re.sub(r"[#>|]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)               # bullets
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)  # emojis
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Feed streamed text with feed(); get back the sentences that are complete so far."""

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out, start = [], 0
        for m in _BOUNDARY.finditer(self.buf):
            piece = self.buf[start:m.end()]
            last_word = re.findall(r"([A-Za-z.]+)[.!?]+[\"')\]]*\s*$", piece)
            if last_word and last_word[0].lower().rstrip(".") in _ABBREV and "\n" not in m.group():
                continue                                          # "Dr. Smith" is not the end
            sentence = clean_for_speech(piece)
            if sentence:
                out.append(sentence)
            start = m.end()
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest, self.buf = clean_for_speech(self.buf), ""
        return [rest] if rest else []


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------
_PYTTSX3_SCRIPT = (
    "import sys, pyttsx3; e = pyttsx3.init(); e.setProperty('rate', int(sys.argv[2])); "
    "e.say(sys.argv[1]); e.runAndWait()"
)


def pick_backend(name: str = "auto") -> str:
    if name != "auto":
        return name
    if sys.platform == "darwin":
        return "say"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    return "pyttsx3"


class Speaker:
    """Speaks sentences in order on a background thread. stop() silences it immediately."""

    def __init__(self, cfg: dict, on_start: Callable[[], None] | None = None,
                 on_done: Callable[[], None] | None = None, enabled: bool = True):
        t = cfg["tts"]
        self.backend, self.voice, self.rate = pick_backend(t.get("backend", "auto")), t.get("voice"), int(t.get("rate", 190))
        self.enabled = enabled
        self.on_start, self.on_done = on_start, on_done
        self._q: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._generation = 0           # bumped by stop(); stale sentences are dropped
        self._busy = False
        self.last_done = 0.0           # time.monotonic() when speech last finished
        self.first_audio_at: float | None = None   # time.perf_counter() of the first sentence handed to the voice
        self._splitter = SentenceSplitter()
        threading.Thread(target=self._worker, name="speaker", daemon=True).start()

    # -- streaming interface --------------------------------------------------
    def feed(self, delta: str) -> None:
        """Give streamed reply text; complete sentences start speaking right away."""
        for s in self._splitter.feed(delta):
            self.say(s)

    def flush(self) -> None:
        for s in self._splitter.flush():
            self.say(s)

    def say(self, sentence: str) -> None:
        sentence = clean_for_speech(sentence)
        if sentence:
            with self._lock:
                self._busy = True
            self._q.put((self._generation, sentence))

    def reset_timer(self) -> None:
        self.first_audio_at = None

    @property
    def speaking(self) -> bool:
        return self._busy

    def wait(self, timeout: float = 60) -> None:
        """Block until everything queued has been spoken (or stopped)."""
        end = time.monotonic() + timeout
        while self.speaking and time.monotonic() < end:
            time.sleep(0.02)

    def stop(self) -> None:
        """Cut speech off now and drop everything queued."""
        with self._lock:
            self._generation += 1
            self._splitter = SentenceSplitter()
            proc = self._proc
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if proc and proc.poll() is None:
            proc.kill()
        self._set_idle()

    # -- internals ------------------------------------------------------------
    def _command(self, text: str) -> list[str]:
        if self.backend == "say":
            cmd = ["say", "-r", str(self.rate)]
            return cmd + (["-v", self.voice] if self.voice else []) + [text]
        if self.backend == "espeak":
            exe = shutil.which("espeak-ng") or "espeak"
            return [exe, "-s", str(self.rate)] + (["-v", self.voice] if self.voice else []) + [text]
        if self.backend == "pyttsx3":
            return [sys.executable, "-c", _PYTTSX3_SCRIPT, text, str(self.rate)]
        raise ValueError(f"unknown tts backend {self.backend!r}")

    def _set_idle(self) -> None:
        with self._lock:
            was_busy, self._busy = self._busy, False
            self.last_done = time.monotonic()
        if was_busy and self.on_done:
            self.on_done()

    def _worker(self) -> None:
        while True:
            gen, text = self._q.get()
            if gen != self._generation:
                continue
            if self.first_audio_at is None:
                self.first_audio_at = time.perf_counter()
            if self.on_start:
                self.on_start()
            log.debug("say: %s", text)
            if self.enabled:
                try:
                    with self._lock:
                        self._proc = subprocess.Popen(self._command(text), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self._proc.wait()
                except Exception as exc:   # a broken voice must never kill this thread
                    log.error("TTS backend %r failed (%s); printing instead: %s", self.backend, exc, text)
                    print("[speaks]", text)
            else:
                print("[speaks]", text)
            if self._q.empty() and gen == self._generation:
                self._set_idle()
