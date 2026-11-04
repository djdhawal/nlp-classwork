"""Streamlit news search over the Qdrant collection built in the notebook (section 2.5).

Run from the NLP folder, after the notebook has closed its Qdrant client:
    conda activate ml_hw
    streamlit run assignment2/search_app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assignment2 import config, vectordb          # noqa: E402
from assignment2.embeddings import load_model     # noqa: E402

SPEC = config.SEARCH_MODELS[config.SEARCH_MODEL]

st.set_page_config(page_title="News Search", layout="wide")
st.title("News Search")
st.caption(f"Qdrant (embedded) · {SPEC['name']} · uint8 scalar-quantized vectors")


@st.cache_resource
def resources():
    return vectordb.get_client(), load_model(SPEC["name"])


client, model = resources()

with st.sidebar:
    st.header("Filters")
    k = st.number_input("Number of results", 5, 50, 10, step=5)
    category = st.selectbox("Category", ["All", *config.LABEL_NAMES.values()])
    source = st.selectbox("Source", ["All", "AG News", "BBC", "Guardian", "Al Jazeera", "CBS News", "NPR", "ABC News"])
    st.write(f"Articles indexed: {client.count(config.COLLECTION).count:,}")

query = st.text_input("Search", placeholder="e.g. oil prices hit record high")
if query:
    t0 = time.perf_counter()
    qv = model.encode(SPEC["query_prefix"] + query, normalize_embeddings=True)
    res = vectordb.search(client, qv, k=int(k),
                          category=None if category == "All" else category,
                          source=None if source == "All" else source)
    st.info(f"{len(res)} results in {(time.perf_counter() - t0) * 1000:.0f} ms (lower distance = closer)")
    for _, r in res.iterrows():
        with st.container(border=True):
            title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
            st.markdown(f"**{title}**  \n`{r['category']}` · {r['source']} · distance {r['distance']}")
            st.write(r["description"])
