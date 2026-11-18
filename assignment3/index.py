"""Section 3.1: chunk the articles, embed the chunks, store them in Qdrant and retrieve the top K.

Qdrant runs in embedded mode (QdrantClient(path=...)), so no Docker or server is needed.
Embedded mode allows one process at a time, so this module keeps one shared client per process.
Call close_store() in the notebook before starting the Streamlit app.
"""

import atexit
import hashlib
import time

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from .config import (
    BATCH_SIZE,
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    DOC_PREFIX,
    EMBED_MODEL,
    QDRANT_PATH,
    QUERY_PREFIX,
    TOP_K,
)

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]  # tried in this order
CHUNK_FIELDS = ["chunk_id", "article_id", "title", "source", "url", "date", "text", "chunk_index"]

_MODEL = None
_CLIENT = None
_QUERY_CACHE = {}


# ---------- chunking ----------
def _merge(pieces, size, overlap):
    """Glue small pieces together into chunks of at most `size` characters.
    When a chunk is full, keep its last ~`overlap` characters as the start of the next one."""
    out, cur, total = [], [], 0
    for p in pieces:
        if cur and total + len(p) > size:
            out.append("".join(cur).strip())
            while cur and (total > overlap or total + len(p) > size):
                total -= len(cur.pop(0))
        cur.append(p)
        total += len(p)
    if cur:
        out.append("".join(cur).strip())
    return out


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, separators=SEPARATORS):
    """Recursive character splitter (same idea as LangChain's RecursiveCharacterTextSplitter):
    split on paragraphs first, then lines, sentences, clauses, words; a piece that is still
    longer than `size` is split again with the next separator."""
    sep = next(s for s in separators if s == "" or s in text)
    rest = separators[separators.index(sep) + 1 :]
    if sep == "":  # no separator left: cut at fixed width
        step = max(size - overlap, 1)
        return [text[i : i + size] for i in range(0, len(text), step)]
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [
        parts[-1]
    ]  # keep the separator at the end of a piece
    out, small = [], []
    for p in (p for p in pieces if p.strip()):
        if len(p) <= size:
            small.append(p)
        else:
            out += _merge(small, size, overlap)
            small = []
            out += split_text(p, size, overlap, rest)
    out += _merge(small, size, overlap)
    return [c for c in out if c]


def chunk_articles(df, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """One Chunk dict per piece of article text. The title goes in the metadata, not the text."""
    chunks = []
    for row in df.itertuples(index=False):
        for i, piece in enumerate(split_text(str(row.text), size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{row.article_id}-C{i:02d}",
                    "article_id": row.article_id,
                    "title": row.title,
                    "source": row.source,
                    "url": row.url,
                    "date": str(row.date),
                    "text": piece,
                    "chunk_index": i,
                }
            )
    return chunks


def chunk_stats(chunks):
    """Counts and length statistics (characters and words) for a list of chunks."""
    df = pd.DataFrame(chunks)
    chars, words = df["text"].str.len(), df["text"].str.split().str.len()
    per_article = df.groupby("article_id").size()
    return pd.Series(
        {
            "articles": df["article_id"].nunique(),
            "chunks": len(df),
            "chunks_per_article_mean": per_article.mean(),
            "chunks_per_article_max": per_article.max(),
            "single_chunk_articles": int((per_article == 1).sum()),
            "chars_mean": chars.mean(),
            "chars_median": chars.median(),
            "chars_min": chars.min(),
            "chars_max": chars.max(),
            "words_mean": words.mean(),
        }
    ).round(1)


# ---------- embeddings ----------
def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    return _MODEL


def embed_texts(texts, is_query=False):
    """L2-normalised float32 embeddings with the Assignment 2 search model (bge-small).
    Queries get the bge query prefix. Document embeddings are cached on disk, keyed on the model
    and a hash of the texts; query embeddings are cached in memory."""
    texts = list(texts)
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    if is_query:
        missing = [t for t in dict.fromkeys(texts) if t not in _QUERY_CACHE]
        if missing:
            emb = (
                load_model()
                .encode(
                    [prefix + t for t in missing],
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                .astype(np.float32)
            )
            _QUERY_CACHE.update(zip(missing, emb))
        return np.stack([_QUERY_CACHE[t] for t in texts])
    digest = hashlib.sha1((prefix + "\x1f".join(texts)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{EMBED_MODEL.replace('/', '__')}__{digest}.npy"
    if path.exists():
        return np.load(path)
    emb = (
        load_model()
        .encode(
            [prefix + t for t in texts],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        )
        .astype(np.float32)
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


# ---------- vector store ----------
def get_store():
    """The shared embedded-Qdrant client (opens the existing store on disk the first time)."""
    global _CLIENT
    if _CLIENT is None:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _CLIENT = QdrantClient(path=str(QDRANT_PATH))
        atexit.register(close_store)
    return _CLIENT


def close_store():
    """Release the on-disk lock so another process (e.g. Streamlit) can open the store."""
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def build_store(chunks, collection=COLLECTION, batch=256):
    """(Re)create `collection` with one point per chunk: vector = chunk embedding (cosine),
    payload = the full Chunk (text + title, article_id, source, url, date, chunk_index)."""
    emb = embed_texts([c["text"] for c in chunks])
    client = get_store()
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=emb.shape[1], distance=models.Distance.COSINE),
    )
    for s in range(0, len(chunks), batch):
        client.upsert(
            collection,
            points=[
                models.PointStruct(id=i, vector=emb[i].tolist(), payload=chunks[i])
                for i in range(s, min(s + batch, len(chunks)))
            ],
        )
    return client


def store_preview(client=None, collection=COLLECTION, n=3):
    """A few stored records (payload + vector size) to show what the store holds."""
    client = client or get_store()
    points, _ = client.scroll(collection, limit=n, with_payload=True, with_vectors=True)
    return pd.DataFrame(
        [
            {
                "point_id": p.id,
                "vector_dim": len(p.vector),
                "vector[:3]": np.round(p.vector[:3], 4).tolist(),
                **{k: p.payload[k] for k in CHUNK_FIELDS if k != "text"},
                "text": p.payload["text"][:120] + "...",
            }
            for p in points
        ]
    )


def store_info(client=None, collection=COLLECTION):
    client = client or get_store()
    info = client.get_collection(collection)
    return {
        "collection": collection,
        "points": client.count(collection).count,
        "dim": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
        "path": str(QDRANT_PATH),
    }


# ---------- retrieval ----------
def retrieve(query, k=TOP_K, client=None, collection=COLLECTION):
    """Top-k chunks for a query. Each Hit = Chunk fields + score (cosine) + rank (1-based)."""
    client = client or get_store()
    q = embed_texts([query], is_query=True)[0]
    res = client.query_points(collection, query=q.tolist(), limit=k, with_payload=True)
    return [{**p.payload, "score": float(p.score), "rank": r} for r, p in enumerate(res.points, 1)]


def hits_table(hits, preview=160):
    """Compact view of hits: rank, score, chunk id, title, source and the start of the text."""
    cols = ["rank", "score", "dense_score", "chunk_id", "title", "source", "text"]
    df = pd.DataFrame(hits)
    if df.empty:
        return df
    df = df[[c for c in cols if c in df.columns]].copy()
    df["text"] = df["text"].str.slice(0, preview).str.replace("\n", " ") + "..."
    return df.round(4).set_index("rank")


def timed_retrieve(query, k=TOP_K, client=None, collection=COLLECTION):
    """retrieve() plus the latency in milliseconds."""
    t0 = time.perf_counter()
    hits = retrieve(query, k, client, collection)
    return hits, round((time.perf_counter() - t0) * 1000, 1)
