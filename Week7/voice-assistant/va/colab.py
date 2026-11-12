"""Google Colab helpers. Colab runs on a server with no microphone or speaker, so audio goes
through the browser: record a clip with JavaScript, decode it with ffmpeg, then hand it to the
same segmenter, speech to text and wake word as on a laptop. Replies play in the cell output
(tts.backend: notebook).
"""
from __future__ import annotations

import base64
import subprocess
import time

import numpy as np

_RECORD_JS = """
async function vaRecord(ms) {
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  const rec = new MediaRecorder(stream), chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  const note = document.createElement('div');
  note.style = 'font: 600 18px sans-serif; color: white; background: #16a34a; padding: 8px 14px; border-radius: 8px; display: inline-block';
  document.body.appendChild(note);
  rec.start();
  const t0 = Date.now();
  const timer = setInterval(() => { note.textContent = '🎙️ Listening… ' + Math.max(0, (ms - (Date.now() - t0)) / 1000).toFixed(1) + ' s'; }, 100);
  await new Promise(r => setTimeout(r, ms));
  rec.stop();
  await new Promise(r => rec.onstop = r);
  clearInterval(timer); note.remove();
  stream.getTracks().forEach(t => t.stop());
  const bytes = new Uint8Array(await new Blob(chunks).arrayBuffer());
  let s = ''; for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return btoa(s);
}
"""


def record_clip(seconds: float = 6.0, sample_rate: int = 16000) -> np.ndarray:
    """Record from the browser's microphone. Allow microphone access when the browser asks."""
    from google.colab.output import eval_js
    from IPython.display import Javascript, display
    display(Javascript(_RECORD_JS))
    data = base64.b64decode(eval_js(f"vaRecord({int(seconds * 1000)})"))
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                          "-ar", str(sample_rate), "pipe:1"], input=data, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def listen_once(assistant, seconds: float = 6.0) -> dict:
    """One push-to-talk turn: record, cut out the speech, transcribe, wake word, then wait for the reply."""
    from .speech_in import Segmenter
    started = time.monotonic()
    clip = record_clip(seconds, assistant.cfg["audio"]["sample_rate"])
    parts = Segmenter(assistant.cfg).split(clip)
    audio = np.concatenate(parts) if parts else clip
    n_turns = len(assistant.turns)
    text = assistant.handle_audio(audio, started_at=started)
    heard = assistant.heard[-1] if assistant.heard and assistant.heard[-1]["text"] == text else {"text": text, "action": "nothing heard"}
    turn = None
    if len(assistant.turns) > n_turns:                       # it was a command: wait for the reply
        turn = assistant.turns[-1]
        while turn["state"] not in ("done", "error"):
            time.sleep(0.05)
        assistant.speaker.wait()
    return {"heard": text, "action": heard["action"], "utterances": len(parts),
            "reply": turn["reply"] if turn else None, "tools": [e["tool"] for e in turn["tools"]] if turn else []}
