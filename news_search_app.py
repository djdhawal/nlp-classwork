import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set(TfidfVectorizer(stop_words="english").get_stop_words()) | {"said", "says", "mr", "mrs"}


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:['-][a-z]+)*", text.lower())


def stem(w):
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    return w


def preprocess(text):
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b, self.N = k1, b, len(corpus)
        self.tf = [Counter(d) for d in corpus]
        self.dl = [len(d) for d in corpus]
        self.avgdl = sum(self.dl) / self.N
        dfc = Counter(t for d in corpus for t in set(d))
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in dfc.items()}

    def scores(self, q):
        out = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
            out[i] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


@st.cache_data
def load_articles():
    return pd.read_csv("news_clean.csv").fillna("")


@st.cache_resource
def build_indexes(texts):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    bm25 = BM25([preprocess(t) for t in texts])
    return tfidf, X, bm25


@st.cache_resource
def load_glove():
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-100")


def embed(wv, text):
    vecs = [wv[t] for t in tokenize(text) if t in wv.key_to_index and t not in STOPWORDS]
    return np.mean(vecs, axis=0) if vecs else np.zeros(wv.vector_size)


def search(method, query, df, tfidf, X, bm25):
    if method == "TF-IDF":
        return cosine_similarity(tfidf.transform([query]), X).ravel()
    if method == "BM25":
        return bm25.scores(preprocess(query))
    wv = load_glove()
    D = np.vstack([embed(wv, t) for t in df["text_clean"]])
    q = embed(wv, query)
    if not q.any():
        return np.zeros(len(df))
    return cosine_similarity(q[None, :], D).ravel()


def show_results(method, scores, df, k):
    st.subheader(method)
    order = np.argsort(scores)[::-1][:k]
    if scores[order[0]] <= 0:
        st.info("No matching articles.")
        return
    for rank, i in enumerate(order, 1):
        row = df.iloc[i]
        st.markdown(f"**{rank}. [{row['title']}]({row['url']})**  \n"
                    f"{row['source']} · {row['date']} · score `{scores[i]:.3f}`")
        st.caption(row["text_clean"][:250] + "...")


st.set_page_config(page_title="News Search", layout="wide")
st.title("News Article Search")

df = load_articles()
tfidf, X, bm25 = build_indexes(tuple(df["text_clean"]))

query = st.text_input("Search query", "climate change")
c1, c2 = st.columns([3, 1])
methods = c1.multiselect("Ranking method(s)", ["TF-IDF", "BM25", "Word2Vec"], default=["TF-IDF", "BM25", "Word2Vec"])
k = c2.slider("Results", 1, 10, 5)

if query and methods:
    cols = st.columns(len(methods))
    for col, m in zip(cols, methods):
        with col:
            show_results(m, search(m, query, df, tfidf, X, bm25), df, k)
