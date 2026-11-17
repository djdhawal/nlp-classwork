"""Section 3.2 step 6: one chat function for every LLM, with a disk cache.

Every provider (Ollama, OpenAI, OpenRouter) speaks the OpenAI chat API, so the model is just
a string "provider:name", e.g. "ollama:llama3.2:3b" or "openai:gpt-4o-mini". Replies are cached
on disk, keyed on model + messages + temperature + max_tokens, so re-running the notebook after
a Colab disconnect costs nothing.
"""
import hashlib
import json
import os
import re
import threading
import time

from . import config

CACHE_FILE = config.CACHE_DIR / "llm_cache.json"
_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_clients = {}
_cache = None
_lock = threading.Lock()


def parse_model(model):
    """'ollama:llama3.2:3b' -> ('ollama', 'llama3.2:3b'). Split on the first colon only."""
    provider, sep, name = model.partition(":")
    if not sep or provider not in config.PROVIDERS:
        raise ValueError(f"Model must look like 'provider:name' with provider in "
                         f"{list(config.PROVIDERS)}; got {model!r}")
    return provider, name


def get_client(provider):
    """OpenAI client pointed at the provider's base_url. Fails early with a readable message."""
    if provider in _clients:
        return _clients[provider]
    from openai import OpenAI

    spec = config.PROVIDERS[provider]
    if spec["key_env"]:
        key = os.environ.get(spec["key_env"])
        if not key:
            raise RuntimeError(f"{spec['key_env']} is not set. Export it (or add it as a Colab secret) "
                               f"to use '{provider}' models.")
    else:
        key = "ollama"  # Ollama ignores the key but the client needs one
        _check_ollama()
    _clients[provider] = OpenAI(base_url=spec["base_url"], api_key=key, timeout=300)
    return _clients[provider]


def _check_ollama():
    import requests

    try:
        requests.get(config.OLLAMA_URL + "/api/version", timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama is not reachable at {config.OLLAMA_URL}. Start it with "
                           f"`ollama serve` (and `ollama pull <model>`). Error: {e}") from None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    return _cache


def _save_cache():
    config.ensure_dirs()
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, indent=1))
    tmp.replace(CACHE_FILE)  # atomic, so a disconnect mid-write cannot corrupt the cache


def _key(model, messages, temperature, max_tokens):
    blob = json.dumps([model, messages, temperature, max_tokens], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def strip_think(text):
    """Remove <think>...</think> reasoning blocks that reasoning models (qwen3, deepseek-r1) emit."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:  # opening tag cut off
        text = text.split("</think>", 1)[1]
    return text.strip()


def chat(messages, model=config.LLM, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
         use_cache=True):
    """Send chat messages to `model` ("provider:name") and return the reply text."""
    provider, name = parse_model(model)
    key = _key(model, messages, temperature, max_tokens)
    with _lock:
        cache = _load_cache()
        if use_cache and key in cache:
            return cache[key]
    client = get_client(provider)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=name, messages=messages, temperature=temperature,
                                                  max_tokens=max_tokens, seed=config.SEED)
            break
        except Exception as e:
            msg = str(e)
            if "not found" in msg.lower() and provider == "ollama":
                raise RuntimeError(f"Ollama model '{name}' is not pulled. Run `ollama pull {name}`.") from None
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    text = strip_think(resp.choices[0].message.content)
    with _lock:
        _cache[key] = text
        _save_cache()
    return text


NO_RAG_SYSTEM = "You are a helpful assistant. Answer the question concisely in 2-4 sentences."


def generate(question, model=config.LLM, **kw):
    """Step 6: plain generation with no retrieval. The model only has what it learned in training."""
    messages = [{"role": "system", "content": NO_RAG_SYSTEM}, {"role": "user", "content": question}]
    return chat(messages, model=model, **kw)


def generate_table(questions, models=config.COMPARE_LLMS, **kw):
    """Ask every model every question (no retrieval). Returns a DataFrame: question x model."""
    import pandas as pd

    rows = []
    for q in questions:
        row = {"question": q}
        for m in models:
            row[m] = generate(q, model=m, **kw)
        rows.append(row)
    return pd.DataFrame(rows)
