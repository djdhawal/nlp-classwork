"""Section 3.3 B: re-rank retrieved chunks with a cross-encoder.

The bi-encoder (bge-small) embeds query and chunk separately; the cross-encoder reads the
query and the chunk together, which is slower but more accurate. The caller retrieves more
candidates than it needs (e.g. 4 * k) and keeps the best k after re-ranking.
"""

from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL, TOP_K
from .index import DEVICE

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(RERANK_MODEL, device=DEVICE)
    return _MODEL


def rerank(query, hits, k=TOP_K):
    """Score every (query, chunk text) pair and return the best k hits, re-numbered.
    `score` becomes the cross-encoder score (a relevance logit, not a cosine; higher = better);
    the original cosine similarity and rank are kept as `dense_score` and `dense_rank`."""
    if not hits:
        return []
    scores = load_model().predict([(query, h["text"]) for h in hits], show_progress_bar=False)
    out = [
        {
            **h,
            "dense_score": h.get("dense_score", h["score"]),
            "dense_rank": h.get("dense_rank", h["rank"]),
            "score": float(s),
        }
        for h, s in zip(hits, scores)
    ]
    out.sort(key=lambda h: h["score"], reverse=True)
    return [{**h, "rank": r} for r, h in enumerate(out[:k], 1)]
