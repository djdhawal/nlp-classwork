"""Section 3.2 step 9: retrieval + generation in one real-time RAG flow.

    query -> input guard -> retrieve (k, or 3k when re-ranking) -> retrieval guard
          -> [re-rank to k] -> grounded prompt -> LLM -> output guard -> answer + citations
"""
import time

from . import config, guardrails, llm, prompts

REFUSAL = prompts.NOT_FOUND
OVERSAMPLE = 3  # with rerank=True, retrieve k*3 candidates and let the cross-encoder keep the best k


def _blocked(reason):
    return f"I can't answer this request ({reason})."


def answer(question, k=config.TOP_K, model=config.LLM, guard=True, rerank=False, strict=False,
           collection=config.COLLECTION, hits=None):
    """Answer `question` from the news corpus.

    Returns a dict: question, answer, hits, citations, guard, latency_s, model (plus raw_answer,
    the LLM text before the output guard). `hits` can be passed in to skip retrieval (tests, eval).
    With guard=True a failed guard replaces the answer with a refusal; with guard=False the guards
    still run and are reported, but nothing is blocked.
    """
    t0 = time.time()
    report = {}
    out = {"question": question, "answer": None, "raw_answer": None, "hits": [], "citations": [],
           "guard": report, "latency_s": None, "model": model}

    def done(text):
        out["answer"] = text
        out["citations"] = prompts.extract_citations(text)
        out["latency_s"] = round(time.time() - t0, 2)
        return out

    # 1. input guard
    g = guardrails.check_input(question)
    report["input"] = g
    query = g.details.get("query", question)
    if guard and not g.ok:
        return done(_blocked(g.reason))
    if not guard:
        query = question

    # 2. retrieval (+ optional re-rank)
    if hits is None:
        from .index import retrieve
        hits = retrieve(query, k=k * OVERSAMPLE if rerank else k, collection=collection)
    g = guardrails.check_retrieval(hits)
    report["retrieval"] = g
    if guard and not g.ok:
        out["hits"] = hits[:k]
        return done(REFUSAL)
    if rerank:
        from .rerank import rerank as rerank_fn
        hits = rerank_fn(query, hits, k=k)
    else:
        hits = hits[:k]
    out["hits"] = hits
    prompt_hits = guardrails.clean_hits(hits) if guard else hits

    # 3. generation
    raw = llm.chat(prompts.build_messages(query, prompt_hits, strict=strict), model=model)
    out["raw_answer"] = raw

    # 4. output guard
    g = guardrails.check_output(raw, hits)
    report["output"] = g
    if guard and not g.ok:
        return done(f"{REFUSAL} (The draft answer was withheld: {g.reason}.)")
    if guard and g.details["refusal"]:
        return done(REFUSAL)  # normalise free-text abstentions to the one fixed sentence
    return done(raw)


def guard_summary(result):
    """One short string per guard, e.g. 'input: ok | retrieval: ok | output: ok'."""
    return " | ".join(f"{name}: {'ok' if g.ok else 'FAIL - ' + g.reason}" for name, g in result["guard"].items())


def results_table(results):
    """DataFrame view of several answer() results."""
    import pandas as pd

    return pd.DataFrame([{"question": r["question"], "answer": r["answer"],
                          "citations": ", ".join(r["citations"]),
                          "top_cosine": round(max(guardrails.dense_score(h) for h in r["hits"]), 3) if r["hits"] else None,
                          "guards": guard_summary(r), "latency_s": r["latency_s"]} for r in results])
