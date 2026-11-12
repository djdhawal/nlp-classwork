# Loop: a local voice assistant (Exercise 7)

Say "Loop, book a dentist appointment tomorrow at 3 PM in Chicago". It books it, answers out loud, and remembers things about you. Everything runs on your laptop.

```
mic -> segmenter (volume) -> faster-whisper -> wake word -> agent (Ollama, tools + memory) -> sentence splitter -> say / espeak
                                                              |
                                              SQLite appointments, Qdrant memory
```

The notebook [`../Exercise-7-Voice-Assistant.ipynb`](../Exercise-7-Voice-Assistant.ipynb) builds this folder stage by stage and runs each check.

## Setup

```bash
ollama pull gemma4:12b-mlx      # or any chat model with tool calling; set llm.model in config.yaml
ollama pull qwen3:8b            # second model for Stage 6
ollama pull nomic-embed-text
docker compose -f ../../Week6/docker-compose.yml up -d     # Qdrant on :6333 (or set memory.qdrant_path)
pip install -r requirements.txt
```

Linux also needs `sudo apt install espeak-ng portaudio19-dev`. On macOS, allow microphone access for the Terminal (or VS Code / Jupyter) app under System Settings, Privacy & Security, Microphone.

## Run

```bash
streamlit run app.py        # web page + always-on listening
python cli.py --voice       # the same, in the terminal
python cli.py               # type commands, hear replies
python checks.py stage1     # the eight typed checks (also stage2, stage4, stage6)
```

All settings (models, wake word and its spellings, audio thresholds, paths) are in `config.yaml`. Logs go to `logs/assistant.log`; every transcript is logged there, which is where new wake-word spellings come from.

## Files

| File | What it does |
|---|---|
| `config.yaml` | every setting |
| `va/appointments.py` | SQLite store behind the four appointment tools |
| `va/agent.py` | system prompt, tool schemas, the streaming tool-calling loop, guards |
| `va/speech_out.py` | markdown clean-up, streaming sentence splitter, `Speaker` with instant `stop()` |
| `va/speech_in.py` | noise-calibrated segmenter, microphone, faster-whisper / mlx-whisper |
| `va/wake.py` | loose wake-word matching |
| `va/memory.py` | Qdrant memory with Ollama embeddings, background fact extraction |
| `va/assistant.py` | ties it together: loader, listener and worker threads, state for the page |
| `app.py` | Streamlit page |
| `cli.py` | terminal front end |
| `checks.py` | the checks for Stages 1, 2, 4 and 6 |

## Design notes

- **Agent.** A small loop over `ollama.Client.chat(..., tools=TOOLS, stream=True)` instead of a framework, so any Ollama tool model works and the page can show each tool call. Every request rebuilds the system prompt with the current date and a list of the next seven dates, so "Friday" maps to a real date without the model doing calendar arithmetic.
- **Guards in code, not only in the prompt** (`llm.guards`). `create_appointment` refuses a location the user never said; `delete_appointment` refuses unless the user's latest message is a yes to a question; an identical booking in the same conversation returns the existing id. When a guard refuses, the tool returns `ok: false` with an instruction, and the model asks the user.
- **Speech.** Reply text streams into a sentence splitter; each finished sentence goes to a TTS child process right away. `stop()` kills the process. The mic is closed while thinking and speaking (plus 0.35 s for echo) and the audio is thrown away.
- **Wake word.** The wake word counts only at the start of a sentence (after optional fillers like "okay" or "hey") or as a vocative between commas. Whisper gets an `initial_prompt` containing "Loop", which makes it spell the name correctly much more often.
- **Memory.** Searched before each request (top 4 above a similarity threshold). Facts are extracted after the reply has been spoken, in a background thread, and only when the user said something about themselves. Anything that looks like a booking is filtered out.

## Stage 3: spoken commands that failed

Copy lines from `logs/assistant.log` (`heard ...`) for every spoken command that did not work.

| # | What I said | What speech-to-text produced | What went wrong |
|---|---|---|---|
| | | | |

## Stage 6: measurements

Filled in by `python checks.py stage6` (or the last section of the notebook) from real runs.

<!-- stage6 -->
| Model | Checks passed (of 8) | Total seconds for the eight checks |
|---|---|---|
| (your default model) | | |
| (one other model) | | |
<!-- /stage6 -->

**Biggest source of mistakes:** _(fill in after your runs, e.g. "the small model invented a location for the haircut and deleted without asking")_

**What I changed to fix it:** _(e.g. "moved those rules from the prompt into the tools, which now refuse and tell the model to ask")_
