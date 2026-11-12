# Week 7: Agent Foundations and Agentic RAG

Agents built with [Strands Agents](https://strandsagents.com) on a local Ollama model. All settings (model, URL, temperature, paths) live in [config.yaml](config.yaml). Install with `pip install -r requirements.txt`.

## RecipeBot: a task-specific agent

RecipeBot is a small cooking assistant. It recommends recipes from the ingredients you have and answers cooking questions. It is the example of a **task-specific agent**: one model, one focused system prompt, one tool.

* **Tool:** `websearch`, a DuckDuckGo search through `ddgs`. The agent calls it to find recipes or look up cooking facts. This is the only part that needs internet access. The model runs locally.
* **Model:** built from `config.yaml` (`ollama.url`, `ollama.models[0]`, `strands.temperature`, `strands.num_ctx`), the same way as the notebooks.
* **Interface:** an interactive terminal loop. Type `exit` to quit.

Code: [recipe-bot-cli/recipe_bot.py](recipe-bot-cli/recipe_bot.py)

```bash
ollama serve                      # if not already running
cd week7/recipe-bot-cli
python recipe_bot.py
```

Example prompts:

* `Suggest a recipe with chicken and broccoli.`
* `How do I cook quinoa?`
* `How can I substitute white wine in shrimp pasta?`

## Appointment assistant

A second CLI agent, with several custom `@tool` functions backed by SQLite: `create_appointment`, `list_appointments`, `update_appointment` and `current_time`. The database path is `tools.appointments_db` in `config.yaml`. Tool errors (a bad date, an unknown id) are raised as exceptions, so the model sees them and can retry.

Code: [appointment-cli/appointment_agent.py](appointment-cli/appointment_agent.py)

```bash
cd week7/appointment-cli
python appointment_agent.py
```

Example: `Book 'Dentist' tomorrow at 3pm in NYC.`, then `Move it to DC.`. Read [7.2-Agent-Tools-MCP.ipynb](7.2-Agent-Tools-MCP.ipynb) for how tools and MCP work.

Read [7.1-Agent-Intro.ipynb](7.1-Agent-Intro.ipynb) first. It covers the pieces RecipeBot is made of: the `Agent`, the `@tool` decorator, and the model from `config.yaml`.

## Memory agent

A third CLI agent with long-term memory in mem0 on embedded Qdrant. Run `python memory_agent.py [user_id]` from `memory-agent-cli/`. Try `Remember that I prefer tea over coffee`, then later `What do I prefer to drink?`. Embedded Qdrant allows one process at a time, so stop the 7.3 notebook first or set `MEMORY_QDRANT_PATH` to another folder.

## Appointment assistant in class

The appointment assistant is the class demo: run it live, watch which tools the model calls, and see how tool errors (a bad date, an unknown id) are fed back to the model.

## Exercise

[Exercise-7.1.md](Exercise-7.1.md): build an always-on voice assistant in six stages (typed appointments agent, speech out, speech in with a wake word, memory, a Streamlit page, measurement). The instructor's reference is `../projects/voice-assistant/`. The earlier exercises are in `../archive/week7/`.

## Contents

| File | Topic |
|---|---|
| `7.1-Agent-Intro.ipynb` | Agent, tools, direct tool calls, logging, model provider |
| `7.2-Agent-Tools-MCP.ipynb` | A simple `@tool`, tool errors, and an MCP server over stdio |
| `7.3-Agent-Memory.ipynb` | Short-term sessions and long-term memory with mem0 |
| `7.4-Agent-Pandas.ipynb` | Data-analyst agent (LangChain) |
| `7.5-Agent-Database.ipynb` | Text-to-SQL agent (LangChain) |
| `7.6-Agentic-RAG.ipynb` | Naive vs corrective vs agentic RAG on the week 6 HotpotQA corpus (needs the week 6 Qdrant collection) |
| `Exercise-7.1.md` | The exercise: build a voice assistant |
| `recipe-bot-cli/` | RecipeBot, the CLI agent described above |
| `appointment-cli/` | Appointment assistant, a CLI agent with custom SQLite tools |
| `memory-agent-cli/` | Personal assistant with long-term memory (mem0), a CLI agent |
| `archive/` | Earlier notebooks |
