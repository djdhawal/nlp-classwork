"""Section 2.5: quantized embeddings, a Qdrant collection, and search evaluation.

Qdrant runs in embedded mode (QdrantClient(path=...)), so no Docker or server is needed.
Embedded mode allows one process at a time: close the notebook client before starting Streamlit.
"""
import time

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from .config import CACHE_DIR, COLLECTION, QDRANT_PATH

RANGES_FILE = CACHE_DIR / "uint8_ranges.npy"


# ---------- quantization ----------
def calibrate(emb):
    """Per-dimension min/max from a calibration set (the documents)."""
    return np.vstack([emb.min(axis=0), emb.max(axis=0)])


def to_uint8(emb, ranges):
    """Scalar quantization: map each dimension's [min, max] onto 0..255 (1 byte instead of 4)."""
    lo, hi = ranges
    q = (emb - lo) / np.maximum(hi - lo, 1e-12) * 255
    return np.clip(np.rint(q), 0, 255).astype(np.uint8)


def to_binary(emb):
    """Binary quantization: keep only the sign of each dimension, packed 8 dims per byte."""
    return np.packbits(emb > 0, axis=1)


def nbytes(arr):
    return arr.size * arr.itemsize


# ---------- brute-force search at each precision (for measuring quantization loss) ----------
def search_float(Q, D, k):
    return np.argsort(-(Q @ D.T), axis=1)[:, :k]


def search_uint8(Qu, Du, k):
    """Euclidean distance. It ignores the +128 shift, so uint8 ranks like signed int8."""
    Qf, Df = Qu.astype(np.float32), Du.astype(np.float32)
    d = (Qf ** 2).sum(1)[:, None] - 2 * Qf @ Df.T + (Df ** 2).sum(1)[None, :]
    return np.argsort(d, axis=1)[:, :k]


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def search_binary(Qb, Db, k):
    ham = np.stack([_POPCOUNT[np.bitwise_xor(q, Db)].sum(axis=1) for q in Qb])
    return np.argsort(ham, axis=1, kind="stable")[:, :k]


def search_binary_rescore(Qb, Db, Q, D, k, oversample=4):
    """Binary search for k*oversample candidates, then rescore them with float vectors."""
    cand = search_binary(Qb, Db, k * oversample)
    out = []
    for qi, c in enumerate(cand):
        out.append(c[np.argsort(-(D[c] @ Q[qi]))[:k]])
    return np.array(out)


# ---------- metrics ----------
def known_item_metrics(top, k=10):
    """Query i should find document i. Returns MRR@k, Recall@1 and Recall@k."""
    target = np.arange(len(top))[:, None]
    hit = top[:, :k] == target
    rank = np.where(hit.any(1), hit.argmax(1) + 1, np.inf)
    return {"MRR@10": float(np.mean(np.where(np.isfinite(rank), 1 / rank, 0))),
            "Recall@1": float(np.mean(rank == 1)), f"Recall@{k}": float(np.mean(np.isfinite(rank)))}


def overlap_at_k(top, ref):
    """Share of the float32 top-k that the quantized search also returns."""
    return float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(top, ref)]))


def compare_quantization(Q, D, k=10):
    ranges = calibrate(D)
    Qu, Du = to_uint8(Q, ranges), to_uint8(D, ranges)
    Qb, Db = to_binary(Q), to_binary(D)
    ref = search_float(Q, D, k)
    runs = {
        "float32": (lambda: search_float(Q, D, k), nbytes(D)),
        "uint8 (scalar)": (lambda: search_uint8(Qu, Du, k), nbytes(Du)),
        "binary": (lambda: search_binary(Qb, Db, k), nbytes(Db)),
        "binary + float rescore": (lambda: search_binary_rescore(Qb, Db, Q, D, k), nbytes(Db)),
    }
    rows = []
    for name, (fn, size) in runs.items():
        t0 = time.perf_counter()
        top = fn()
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        rows.append({"precision": name, "bytes_per_vector": size // len(D),
                     "index_KB": round(size / 1024, 1), **known_item_metrics(top, k),
                     "overlap@10_vs_float": overlap_at_k(top, ref), "ms_per_query": round(ms, 3)})
    return pd.DataFrame(rows).set_index("precision").round(4)


# ---------- Qdrant ----------
def get_client(path=QDRANT_PATH):
    return QdrantClient(path=str(path))


def build_collection(client, emb_float, payloads, name=COLLECTION, batch=512):
    """Quantize to uint8 with ranges from the corpus, save the ranges (queries need them),
    and upload the vectors with their metadata."""
    ranges = calibrate(emb_float)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(RANGES_FILE, ranges)
    vecs = to_uint8(emb_float, ranges)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(
        size=vecs.shape[1], distance=models.Distance.EUCLID, datatype=models.Datatype.UINT8))
    for s in range(0, len(vecs), batch):
        client.upload_points(name, points=[
            models.PointStruct(id=i, vector=vecs[i].tolist(), payload=payloads[i])
            for i in range(s, min(s + batch, len(vecs)))])
    return client.count(name).count


def search(client, query_vec, k=10, category=None, source=None, name=COLLECTION, ranges=None):
    """query_vec is a float embedding; it is quantized with the stored ranges before searching."""
    ranges = np.load(RANGES_FILE) if ranges is None else ranges
    q = to_uint8(query_vec[None, :], ranges)[0].tolist()
    must = []
    if category:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    res = client.query_points(name, query=q, limit=k, query_filter=models.Filter(must=must) if must else None)
    return pd.DataFrame([{"id": p.id, "distance": round(p.score, 1), **p.payload} for p in res.points])


def category_precision(client, encode_query, queries, k=10):
    """P@k where a hit is a result whose category matches the query's intended category."""
    rows = []
    for q, want in queries:
        t0 = time.perf_counter()
        res = search(client, encode_query(q), k=k)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"query": q, "intended": want, f"P@{k}": (res["category"] == want).mean(),
                     "top_result": res["title"].iloc[0], "latency_ms": round(ms, 1)})
    return pd.DataFrame(rows)
