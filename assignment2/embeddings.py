"""Section 2.3: sentence-encoder features for classification, with on-disk caching."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .classic import scores
from .config import BATCH_SIZE, CACHE_DIR, SEED

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
_MODELS = {}


def load_model(name):
    if name not in _MODELS:
        _MODELS[name] = SentenceTransformer(name, device=DEVICE)
    return _MODELS[name]


def model_info(name):
    m = load_model(name)
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
            "dim": m.get_embedding_dimension(), "max_seq_length": m.max_seq_length}


def encode(name, texts, prefix="", cache_key=None):
    """L2-normalised float32 embeddings. Saved to cache/<model>__<cache_key>.npy when cache_key is given."""
    path = CACHE_DIR / f"{name.replace('/', '__')}__{cache_key}.npy" if cache_key else None
    if path is not None and path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    emb = load_model(name).encode([prefix + t for t in texts], batch_size=BATCH_SIZE,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)
    if path is not None:
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(path, emb)
    return emb


def compare_encoders(models, train, test, scraped):
    """Encode the three text sets with every model, fit LogisticRegression on train embeddings,
    and score on the AG News test split and on the hand-labelled scraped articles."""
    rows, preds = [], {}
    for key, spec in models.items():
        t0 = time.perf_counter()
        F = {split: encode(spec["name"], df["text"].tolist(), spec["prefix"], cache_key=split)
             for split, df in (("train", train), ("test", test), ("scraped", scraped))}
        enc_s = time.perf_counter() - t0
        clf = LogisticRegression(C=2.0, max_iter=3000, random_state=SEED).fit(F["train"], train["label"])
        p_test, p_scr = clf.predict(F["test"]), clf.predict(F["scraped"])
        preds[key] = {"test": p_test, "scraped": p_scr}
        t, s = scores(test["label"], p_test), scores(scraped["label"], p_scr)
        rows.append({"model": key, **model_info(spec["name"]),
                     "test_accuracy": t["accuracy"], "test_macro_F1": t["macro_F1"],
                     "scraped_accuracy": s["accuracy"], "scraped_weighted_F1": s["weighted_F1"],
                     "encode_seconds": enc_s})
    return pd.DataFrame(rows).set_index("model").round(4), preds


def plot_comparison(table, metrics=("test_accuracy", "test_macro_F1", "scraped_accuracy", "scraped_weighted_F1")):
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#e0a100", "#52514e", "#c0392b"]
    metrics = list(metrics)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n = len(table)
    w = 0.8 / n
    x = np.arange(len(metrics))
    for i, (name, c) in enumerate(zip(table.index, colors)):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, table.loc[name, metrics].astype(float), w * 0.95,
                      color=c, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x, metrics)
    lo = float(table[metrics].min().min())
    ax.set_ylim(max(0, lo - 0.1), 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Classification metrics by feature set")
    plt.tight_layout()
