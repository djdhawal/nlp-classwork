"""Task 2: cosine-similarity retrieval over the test reviews, scored with precision@k."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import TOP_K, LABEL_NAMES


def rank(query_vecs, doc_vecs, k=TOP_K):
    sims = cosine_similarity(query_vecs, doc_vecs)
    top = np.argsort(-sims, axis=1)[:, :k]
    return top, np.take_along_axis(sims, top, axis=1)


def evaluate_retrieval(encode_queries, doc_vecs, y_docs, queries, k=TOP_K):
    texts = [q for q, _ in queries]
    wants = np.array([w for _, w in queries])
    top, scores = rank(encode_queries(texts), doc_vecs, k)
    hits = (y_docs[top] == wants[:, None])
    per_query = pd.DataFrame({
        "query": texts,
        "wants": [LABEL_NAMES[w] for w in wants],
        f"P@{k}": hits.mean(axis=1),
    })
    return {"mean_p_at_k": hits.mean(), "per_query": per_query, "top": top, "scores": scores,
            "doc_vecs": doc_vecs}


def top_result(result, query_idx, docs, y_docs, chars=400):
    """Top-1 review for a query, for qualitative inspection."""
    d = result["top"][query_idx, 0]
    return {
        "score": float(result["scores"][query_idx, 0]),
        "label": LABEL_NAMES[int(y_docs[d])],
        "text": docs[d][:chars].replace("<br />", " ") + ("..." if len(docs[d]) > chars else ""),
    }
