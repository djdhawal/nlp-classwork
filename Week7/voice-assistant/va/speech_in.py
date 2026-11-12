"""Speech in: microphone capture, utterance segmentation by volume, and local speech to text."""
from __future__ import annotations

import collections
import logging
import queue
import time

import numpy as np

log = logging.getLogger("va.stt")


class Segmenter:
    """Cut a stream of audio frames into utterances using the room's noise level.

    feed(frame) returns a finished utterance (float32 array) or None. Pure numpy, so it can be
    tested without a microphone.
    """

    def __init__(self, cfg: dict):
        a = cfg["audio"]
        self.sr = a["sample_rate"]
        self.frame_s = a["frame_ms"] / 1000
        self.factor, self.min_threshold = a["threshold_factor"], a["min_threshold"]
        self.calib_frames = int(a["calibration_seconds"] / self.frame_s)
        self.preroll_frames = int(a["preroll_seconds"] / self.frame_s)
        self.end_frames = int(a["end_silence_seconds"] / self.frame_s)
        self.min_speech_frames = int(a["min_speech_seconds"] / self.frame_s)
        self.max_frames = int(a["max_utterance_seconds"] / self.frame_s)
        self.noise: float | None = None
        self._calib: list[float] = []
        self.reset()

    @property
    def threshold(self) -> float:
        return max((self.noise or 0.0) * self.factor, self.min_threshold)

    @property
    def calibrated(self) -> bool:
        return self.noise is not None

    def reset(self) -> None:
        """Drop any partial utterance (used when the mic is muted). Keeps the noise level."""
        self.preroll: collections.deque = collections.deque(maxlen=max(self.preroll_frames, 1))
        self.frames: list[np.ndarray] = []
        self.in_speech, self.silent_run, self.voiced = False, 0, 0

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        if self.noise is None:                                   # 1) calibrate on the first second
            self._calib.append(rms)
            if len(self._calib) >= self.calib_frames:
                self.noise = float(np.median(self._calib))
                log.info("noise level %.4f -> threshold %.4f", self.noise, self.threshold)
            return None
        loud = rms > self.threshold
        if not self.in_speech:
            if loud:                                             # 2) speech starts: keep the pre-roll
                self.in_speech, self.frames = True, list(self.preroll) + [frame]
                self.voiced, self.silent_run = 1, 0
            else:
                self.preroll.append(frame)
                self.noise = 0.98 * self.noise + 0.02 * rms      # follow slow changes in room noise
            return None
        self.frames.append(frame)
        if loud:
            self.voiced += 1
            self.silent_run = 0
        else:
            self.silent_run += 1
        if self.silent_run >= self.end_frames or len(self.frames) >= self.max_frames:   # 3) ~0.6 s quiet ends it
            audio, voiced = np.concatenate(self.frames), self.voiced
            self.reset()
            if voiced >= self.min_speech_frames:
                return audio
            log.debug("dropped a %.2fs blip", voiced * self.frame_s)
        return None


class Microphone:
    """Continuous capture with sounddevice. Frames go into a queue read by the listener thread."""

    def __init__(self, cfg: dict):
        import sounddevice as sd
        a = cfg["audio"]
        self.sr, self.blocksize = a["sample_rate"], int(a["sample_rate"] * a["frame_ms"] / 1000)
        self.q: queue.Queue = queue.Queue()
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=self.blocksize,
                                     device=a.get("input_device"), callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream.start()
        return self

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Throw away everything recorded so far (e.g. while the assistant was speaking)."""
        n = 0
        while True:
            try:
                self.q.get_nowait()
                n += 1
            except queue.Empty:
                return n


class Transcriber:
    """Local speech to text: faster-whisper anywhere, or mlx-whisper on Apple Silicon."""

    def __init__(self, cfg: dict):
        s = cfg["stt"]
        self.backend, self.model_name = s["backend"], s["model"]
        self.prompt = s.get("initial_prompt")
        self.sr = cfg["audio"]["sample_rate"]
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_name, device="auto", compute_type=s.get("compute_type", "int8"))
        elif self.backend == "mlx-whisper":
            import mlx_whisper
            self.model = mlx_whisper
        else:
            raise ValueError(f"unknown stt backend {self.backend!r}")
        self.transcribe(np.zeros(self.sr // 2, dtype=np.float32))   # warm up

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        t0 = time.perf_counter()
        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(audio, language="en", beam_size=1, initial_prompt=self.prompt,
                                                condition_on_previous_text=False, vad_filter=False)
            text = " ".join(s.text for s in segments)
        else:
            out = self.model.transcribe(audio, path_or_hf_repo=self.model_name, language="en",
                                        initial_prompt=self.prompt, condition_on_previous_text=False)
            text = out["text"]
        text = text.strip()
        log.info("stt %.2fs (%.1fs audio): %r", time.perf_counter() - t0, len(audio) / self.sr, text)
        return text


def beep(cfg: dict, freq: float = 880.0, seconds: float = 0.12) -> None:
    """A short tone that says 'I am listening again'."""
    try:
        import sounddevice as sd
        sr = 22050
        t = np.arange(int(sr * seconds)) / sr
        tone = 0.2 * np.sin(2 * np.pi * freq * t) * np.hanning(t.size)
        sd.play(tone.astype(np.float32), sr, blocking=True)
    except Exception:
        log.debug("beep failed", exc_info=True)
