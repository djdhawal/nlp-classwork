"""Streamlit chatbot over the news RAG pipeline (step 10).

Run from the NLP folder:
    streamlit run assignment3/app.py

Embedded Qdrant allows only one process at a time. If the notebook is open,
run `client.close()` there first, or the app cannot open the collection.
On Colab use `assignment3.colab_app.launch()` instead.

Set A3_FAKE_RAG=1 to run the UI with canned answers (no Qdrant, no LLM).
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assignment3 import config  # noqa: E402

FAKE = os.environ.get("A3_FAKE_RAG") == "1"
CITE = re.compile(r"\[\s*([A-Z]\d{4}-C\d+)\s*(?:\|[^\]]*)?\]")


def fake_answer(
    question, k=config.TOP_K, model=config.LLM, guard=True, rerank=False, strict=False
):
    """Canned reply with the same keys as rag.answer, for UI testing."""
    if guard and "ignore previous" in question.lower():
        return {
            "question": question,
            "answer": "",
            "hits": [],
            "citations": [],
            "model": model,
            "guard": {"ok": False, "reason": "blocked: prompt injection"},
            "latency_s": 0.01,
        }
    hits = [
        {
            "chunk_id": f"A000{i}-C01",
            "article_id": f"A000{i}",
            "title": f"Example headline {i}",
            "source": "BBC",
            "url": "https://www.bbc.com/news",
            "date": "2026-11-10",
            "text": "Example passage text about the news story. " * 6,
            "chunk_index": 1,
            "score": 0.8 - 0.05 * i,
            "rank": i,
        }
        for i in range(1, k + 1)
    ]
    text = (
        "Central banks held rates steady this week [A0001-C01], while oil prices fell [A0002-C01]."
    )
    return {
        "question": question,
        "answer": text,
        "hits": hits,
        "citations": ["A0001-C01", "A0002-C01"],
        "guard": {"ok": True, "reason": ""},
        "latency_s": 1.23,
        "model": model,
    }


@st.cache_resource(show_spinner="Loading vector store and models...")
def backend():
    """Import the pipeline once per server and open the Qdrant store."""
    if FAKE:
        return fake_answer
    from assignment3 import index, rag

    index.get_store()  # opens embedded Qdrant once; fails here if the notebook still holds it
    return rag.answer


def guard_info(g):
    """Turn the guard field (GuardResult, dict or None) into (ok, reason)."""
    if g is None:
        return True, ""
    if isinstance(g, dict):
        return bool(g.get("ok", True)), str(g.get("reason", ""))
    return bool(getattr(g, "ok", True)), str(getattr(g, "reason", ""))


def md_safe(text):
    """Escape $ so Streamlit does not render '$115.8 billion ... $155' as LaTeX math."""
    return str(text).replace("$", r"\$")


def with_links(text, hits):
    """Make [A0001-C02] and [A0001-C02 | Title] citations clickable links to the article."""
    urls = {h.get("chunk_id"): h.get("url") for h in hits}

    def link(m):
        url = urls.get(m.group(1))
        return f"[[{m.group(0)[1:-1].strip()}]]({url})" if url else m.group(0)

    return CITE.sub(link, md_safe(text))


def show_reply(res):
    """Render one assistant turn: answer, guard note, latency, sources."""
    ok, reason = guard_info(res.get("guard"))
    hits = res.get("hits") or []
    if res.get("answer"):
        st.markdown(with_links(res["answer"], hits))
    if not ok:
        st.warning(f"Guardrail: {reason or 'blocked'}", icon="🛡️")
    elif reason:
        st.info(f"Guardrail note: {reason}")
    cites = res.get("citations") or []
    st.caption(
        f"{res.get('model', '')} · {res.get('latency_s', 0):.2f} s · {len(hits)} chunks · "
        f"cited: {', '.join(cites) if cites else 'none'}"
    )
    if hits:
        with st.expander(f"Sources ({len(hits)})"):
            for h in hits:
                mark = " ✅" if h.get("chunk_id") in cites else ""
                name = md_safe(h.get("title", ""))
                title = f"[{name}]({h['url']})" if h.get("url") else name
                st.markdown(
                    f"**{h.get('rank', '')}. {title}**{mark}  \n"
                    f"`{h.get('chunk_id', '')}` · {h.get('source', '')} · {h.get('date', '')} · "
                    f"score {h.get('score', 0):.3f}"
                )
                st.text(str(h.get("text", ""))[:300] + "...")


st.set_page_config(page_title="News RAG Chat", page_icon="📰")
st.title("News RAG Chat")
st.caption(
    f"Qdrant · {config.EMBED_MODEL} · answers cite chunk ids"
    + (" · DEMO MODE (fake answers)" if FAKE else "")
)

with st.sidebar:
    st.header("Settings")
    models = list(dict.fromkeys([config.LLM, *config.COMPARE_LLMS]))
    model = st.selectbox("Model", models)
    custom = st.text_input("Other model (provider:name)", placeholder="openai:gpt-4o-mini")
    model = custom.strip() or model
    k = st.slider("Chunks to retrieve (k)", 1, 10, config.TOP_K)
    guard = st.toggle("Guardrails", value=True)
    rerank = st.toggle("Re-rank (cross-encoder)", value=False)
    strict = st.toggle("Strict prompt", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    answer_fn = backend()
except Exception as e:  # most often: Qdrant folder locked by the notebook
    st.error(
        f"Could not load the pipeline: {e}\n\n"
        "If the notebook is running, call `client.close()` there and reload."
    )
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(md_safe(msg["content"]))
        else:
            show_reply(msg["content"])

if question := st.chat_input("Ask about the news..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_safe(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                res = answer_fn(
                    question, k=k, model=model, guard=guard, rerank=rerank, strict=strict
                )
            except Exception as e:
                res = {
                    "answer": "",
                    "hits": [],
                    "guard": {"ok": False, "reason": f"error: {e}"},
                    "latency_s": 0.0,
                    "model": model,
                }
        show_reply(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
