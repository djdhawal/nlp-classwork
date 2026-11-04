"""Shared helpers for the Week 6 RAG notebooks.

Centralises what every retrieval/RAG notebook needs: config loading, the HotpotQA
caches, service checks, the Qdrant connection, LLM clients, answer-quality metrics and
the shared results ledger that lets every notebook add a row to one comparison table.

Usage (notebook running in code/week6):
    import rag_common as rc
    cfg = rc.load_config()
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean

import yaml

HERE = Path(__file__).resolve().parent

RAG_PROMPT_TEXT = """You are a helpful AI assistant that answers questions based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the information in the context to answer.
- If the answer cannot be found in the context, say "The information is not available in the provided context."
- Reply with only the answer itself (a few words, no explanation, no full sentence).

Answer:
"""


# ── Configuration, data and services ─────────────────────────────────────────
def load_config() -> dict:
    """Read config.yaml (next to this file) — the single source of models, URLs and paths."""
    return yaml.safe_load((HERE / "config.yaml").read_text())


def load_hotpotqa(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Return (qa_pairs, passages) cached by 6.5-Storage-Vector-Databases.ipynb."""
    h = cfg["hotpotqa"]
    qa_path, passages_path = Path(h["qa_pairs_cache"]), Path(h["passages_cache"])
    if not (qa_path.exists() and passages_path.exists()):
        raise FileNotFoundError("HotpotQA caches not found. Run 6.5-Storage-Vector-Databases.ipynb first.")
    return json.loads(qa_path.read_text()), json.loads(passages_path.read_text())


def check_services(cfg: dict, qdrant: bool = True, llm: bool = True) -> bool:
    """Say which required service is down and how to start it. Returns True if all are reachable."""
    def reachable(url: str) -> bool:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    ok = True
    if qdrant:
        up = reachable(cfg["qdrant"]["url"] + "/collections")
        print(f"Qdrant  {cfg['qdrant']['url']}: {'OK' if up else 'NOT REACHABLE -> run `docker compose up -d` in code/week6'}")
        ok &= up
    if llm:
        base = cfg["llm"]["base_url"]
        if "localhost" in base or "127.0.0.1" in base:
            up = reachable(cfg["ollama"]["url"] + "/api/tags")
            print(f"Ollama  {cfg['ollama']['url']}: {'OK' if up else 'NOT REACHABLE -> run `ollama serve`, then `ollama pull ' + cfg['llm']['model'] + '`'}")
        else:
            up = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
            print(f"LLM     {base}: {'API key found' if up else 'NO API KEY -> set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env'}")
        ok &= up
    return ok


def eval_set(qa_pairs: list[dict], n: int = 15, seed: int = 76) -> list[dict]:
    """The one fixed question sample every notebook scores on, so results are comparable."""
    import random

    return random.Random(seed).sample(qa_pairs, k=min(n, len(qa_pairs)))


def connect_qdrant(cfg: dict, collection: str | None = None):
    """Connect to an existing Qdrant collection. Returns (client, embeddings, vector_store)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    collection = collection or cfg["hotpotqa"]["collection"]
    client = QdrantClient(url=cfg["qdrant"]["url"])
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Collection '{collection}' not found. Start Qdrant and run 6.5-Storage-Vector-Databases.ipynb first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=cfg["embeddings"]["model"])
    store = QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)
    return client, embeddings, store


def make_llm(cfg: dict, model: str | None = None, temperature: float = 0.0):
    """OpenAI-compatible chat client (Ollama or OpenRouter) built from config.yaml."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "ollama"
    return ChatOpenAI(
        model=model or cfg["llm"]["model"],
        base_url=cfg["llm"]["base_url"],
        api_key=api_key,
        temperature=temperature,
    )


def load_llms(cfg: dict) -> dict:
    """Instantiate every model listed in config.yaml and keep the ones that respond."""
    available = {}
    names = [cfg["llm"]["model"]] + [m for m in cfg["llm"]["models"] if m != cfg["llm"]["model"]]   # default model first
    for name in names:
        llm = make_llm(cfg, name)
        try:
            llm.invoke("Reply with OK.")
            available[name] = llm
            print(f"{name}: available")
        except Exception as exc:
            print(f"{name}: skipped ({str(exc)[:80]})")
    if not available:
        raise RuntimeError("No LLM reachable. Start Ollama (or set an API key) and check llm.* in config.yaml.")
    return available


# ── Prompting ────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def answer_with_docs(question: str, docs, llm) -> str:
    """Generate an answer from pre-retrieved documents."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_template(RAG_PROMPT_TEXT) | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


# ── Metrics ──────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def retrieval_hit(docs, reference: str) -> int:
    """1 if the reference answer appears verbatim in the retrieved context."""
    return int(normalize_text(reference) in normalize_text(format_docs(docs)))


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    overlap = sum((Counter(pred) & Counter(ref)).values()) if pred and ref else 0
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def fuzzy_score(prediction: str, reference: str) -> float:
    from rapidfuzz import fuzz

    return fuzz.token_set_ratio(prediction, reference) / 100.0


def evaluate_answers(name: str, llm, examples: list[dict]) -> list[dict]:
    """Answer each example (dict with question, answer, docs) and score it."""
    from tqdm.auto import tqdm

    rows = []
    for ex in tqdm(examples, desc=name):
        t0 = time.time()
        prediction = answer_with_docs(ex["question"], ex["docs"], llm)
        rows.append({
            "question": ex["question"],
            "reference": ex["answer"],
            "prediction": prediction,
            "contexts": [d.page_content for d in ex["docs"]],
            "num_docs": len(ex["docs"]),
            "total_chars": sum(len(d.page_content) for d in ex["docs"]),
            "exact_match": exact_match(prediction, ex["answer"]),
            "token_f1": token_f1(prediction, ex["answer"]),
            "rapidfuzz": fuzzy_score(prediction, ex["answer"]),
            "retrieval_hit": retrieval_hit(ex["docs"], ex["answer"]),
            "inference_time": time.time() - t0,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Average every numeric column of evaluate_answers() output."""
    keys = ["num_docs", "total_chars", "exact_match", "token_f1", "rapidfuzz", "retrieval_hit", "inference_time"]
    return {k: mean(r[k] for r in rows) for k in keys}


# ── Shared results ledger: one table across the whole RAG course ─────────────
LEDGER = Path("/tmp/rag_ablation.json")


def record_result(label: str, stage: str, rows: list[dict] | None = None, hit: float | None = None,
                  f1: float | None = None, n: int | None = None) -> None:
    """Add (or replace) one row in the shared ledger.

    Pass `rows` from evaluate_answers()/your own loop (keys retrieval_hit/token_f1 or hit/f1),
    or pass hit/f1 directly. `stage` groups rows: chunking, retrieval, post-retrieval, pipeline.
    """
    if rows:
        hit = mean(r.get("retrieval_hit", r.get("hit")) for r in rows)
        f1 = mean(r["token_f1"] if "token_f1" in r else r["f1"] for r in rows) if any(("token_f1" in r or "f1" in r) for r in rows) else None
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    ledger[label] = {"stage": stage, "hit": hit, "f1": f1, "n": len(rows) if rows else n}
    LEDGER.write_text(json.dumps(ledger, indent=1))


def show_ablation() -> None:
    """Print the ledger: every technique tried so far on the same HotpotQA sample."""
    if not LEDGER.exists():
        print("Ledger is empty. Run the retrieval notebooks (6.6-6.9) first.")
        return
    ledger = json.loads(LEDGER.read_text())
    print(f"{'Technique':<34}{'Stage':<16}{'Hit %':>7}{'F1 %':>7}{'n':>5}")
    print("-" * 69)
    for label, r in ledger.items():
        f1 = f"{r['f1'] * 100:>7.1f}" if r["f1"] is not None else f"{'-':>7}"
        n = f"{r['n']:>5}" if r["n"] else f"{'-':>5}"
        print(f"{label:<34}{r['stage']:<16}{r['hit'] * 100:>7.1f}{f1}{n}")
    if any(r["stage"].endswith("*") for r in ledger.values()):
        print("\n* Chunking rows use a 100-passage mini index. Compare them with each other, not with the other rows.")


def reset_ablation() -> None:
    LEDGER.unlink(missing_ok=True)
