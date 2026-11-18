"""Section 3.4 (bonus): BERTopic on the news corpus, a coherence sweep over the number of topics,
and Guided BERTopic with seed words for the four AG News classes, compared with Assignment 2."""

import copy
import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from umap import UMAP

from . import config

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
MAX_DOC_CHARS = 1000  # title + start of the article is enough to find its topic
MIN_TOPIC_SIZE = 5  # small HDBSCAN clusters, so there are enough topics to reduce from
NR_TOPICS_GRID = (5, 8, 10, 15, 20, "auto", None)  # None = keep every HDBSCAN cluster
N_AG_EVAL = 2000  # balanced AG News test sample for the supervised comparison
LABEL_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# Extra stop words: news boilerplate that otherwise tops every topic.
STOP_WORDS = list(
    ENGLISH_STOP_WORDS
    | {
        "said",
        "says",
        "say",
        "new",
        "year",
        "years",
        "just",
        "like",
        "time",
        "people",
        "told",
        "according",
        "reuters",
        "ap",
        "afp",
        "bbc",
        "cnn",
        "cbs",
        "npr",
        "news",
        "quot",
        "lt",
        "gt",
        "href",
        "com",
        "www",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "et",
        "pm",
        "am",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)

# Seed words for Guided BERTopic, one list per AG News class (same order as LABEL_NAMES).
SEED_TOPICS = {
    "World": [
        "government",
        "president",
        "minister",
        "election",
        "war",
        "military",
        "troops",
        "police",
        "killed",
        "attack",
        "peace",
        "leader",
        "country",
        "international",
        "nations",
        "iraq",
    ],
    "Sports": [
        "game",
        "team",
        "season",
        "coach",
        "players",
        "win",
        "league",
        "cup",
        "match",
        "championship",
        "olympic",
        "football",
        "baseball",
        "basketball",
        "tennis",
        "soccer",
    ],
    "Business": [
        "company",
        "market",
        "stocks",
        "shares",
        "profit",
        "sales",
        "oil",
        "prices",
        "economy",
        "bank",
        "billion",
        "investors",
        "earnings",
        "deal",
        "percent",
        "trade",
    ],
    "Sci/Tech": [
        "software",
        "internet",
        "microsoft",
        "computer",
        "technology",
        "space",
        "scientists",
        "research",
        "google",
        "data",
        "users",
        "online",
        "nasa",
        "phone",
        "chip",
        "ai",
    ],
}

# Best supervised results from Assignment 2, copied from the outputs of Assignment-2.ipynb
# (section 2.2 cell "Pick the best model" and section 2.3 encoder table). Both are on the full
# AG News test split (7,600 articles).
A2_SUPERVISED = [
    {
        "method": "TF-IDF + LinearSVC (A2 2.2, best)",
        "accuracy": 0.9208,
        "macro_F1": 0.9206,
        "labels_used": "120k train",
        "eval_set": "AG test (7,600)",
    },
    {
        "method": "bge-small-en-v1.5 + LogReg (A2 2.3)",
        "accuracy": 0.9021,
        "macro_F1": 0.9019,
        "labels_used": "20k train",
        "eval_set": "AG test (7,600)",
    },
]

_ENCODER = {}


# ---------- documents and embeddings ----------


def prepare_docs(df, max_chars=MAX_DOC_CHARS):
    """One string per article: title + start of the text (AG News: title + description)."""
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    docs = [
        (t + ". " + x)[:max_chars] if not x.startswith(t) else x[:max_chars]
        for t, x in zip(title, text)
    ]
    return docs


def _encoder(model_name=config.EMBED_MODEL):
    if model_name not in _ENCODER:
        _ENCODER[model_name] = SentenceTransformer(model_name, device=DEVICE)
    return _ENCODER[model_name]


def embed_docs(docs, model_name=config.EMBED_MODEL):
    """L2-normalised embeddings, cached on disk by a hash of the texts, so sweeps reuse them."""
    key = hashlib.md5("\n".join(docs).encode()).hexdigest()[:12]
    path = config.CACHE_DIR / f"topics_{model_name.replace('/', '__')}_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = (
        _encoder(model_name)
        .encode(
            [config.DOC_PREFIX + d for d in docs],
            batch_size=config.BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        .astype(np.float32)
    )
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


# ---------- BERTopic ----------


def make_model(
    nr_topics=None, seed_topic_list=None, min_topic_size=MIN_TOPIC_SIZE, seed=config.SEED
):
    """BERTopic with fixed-seed UMAP, HDBSCAN and a stop-word CountVectorizer.
    Guided mode needs the encoder itself, to embed the seed-word lists in the same space
    as the docs.
    """
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=STOP_WORDS, min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    encoder = _encoder() if seed_topic_list else None
    return BERTopic(
        embedding_model=encoder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=nr_topics,
        seed_topic_list=seed_topic_list,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )


def fit_topics(
    docs, embeddings, nr_topics=None, seed_topic_list=None, min_topic_size=MIN_TOPIC_SIZE
):
    """Fit BERTopic on precomputed embeddings. Returns (model, topic id per doc; -1 = outlier)."""
    model = make_model(nr_topics, seed_topic_list, min_topic_size)
    topics, _ = model.fit_transform(
        docs, embeddings.copy()
    )  # guided mode edits embeddings in place
    return model, np.asarray(topics)


def topic_table(model, n_words=8):
    """One row per topic: size, top words and the most representative document (shortened)."""
    info = model.get_topic_info()
    info = info[info["Topic"] != -1]
    rows = []
    for _, r in info.iterrows():
        words = [w for w, _ in model.get_topic(r["Topic"])][:n_words]
        rep = (r.get("Representative_Docs") or [""])[0]
        rows.append(
            {
                "topic": r["Topic"],
                "size": r["Count"],
                "top_words": ", ".join(words),
                "example": rep[:90],
            }
        )
    return pd.DataFrame(rows).set_index("topic")


# ---------- coherence ----------


def coherence(model, docs, topn=10, measures=("c_v", "c_npmi")):
    """Topic coherence with gensim, on docs tokenised the same way BERTopic's vectorizer does.
    Outlier topic -1 is left out."""
    analyzer = model.vectorizer_model.build_analyzer()
    tokens = [analyzer(d) for d in docs]
    dictionary = Dictionary(tokens)
    topic_words = []
    for t in sorted(model.get_topics()):
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:topn] if w and w in dictionary.token2id]
        if len(words) >= 2:
            topic_words.append(words)
    out = {}
    for m in measures:
        cm = CoherenceModel(
            topics=topic_words, texts=tokens, dictionary=dictionary, coherence=m, processes=1
        )
        out[m] = cm.get_coherence()
    return out


def sweep_nr_topics(docs, embeddings, grid=NR_TOPICS_GRID, min_topic_size=MIN_TOPIC_SIZE):
    """Fit once (UMAP + HDBSCAN are the slow part), then reduce a copy to each nr_topics value
    and score coherence. Returns a table with one row per setting."""
    base, base_topics = fit_topics(docs, embeddings, None, None, min_topic_size)
    rows = []
    for k in grid:
        t0 = time.perf_counter()
        model = copy.deepcopy(base)
        if k is not None:
            model.reduce_topics(docs, nr_topics=k)
        topics = np.asarray(model.topics_)
        rows.append(
            {
                "nr_topics": "none" if k is None else str(k),
                "n_topics_found": int(len(set(topics) - {-1})),
                "outlier_share": round(float((topics == -1).mean()), 3),
                **{m: round(v, 4) for m, v in coherence(model, docs).items()},
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
    return pd.DataFrame(rows).set_index("nr_topics")


def best_nr_topics(sweep, metric="c_v"):
    """Grid value with the highest coherence, converted back to what BERTopic expects."""
    best = sweep[metric].idxmax()
    return None if best == "none" else ("auto" if best == "auto" else int(best))


def plot_sweep(sweep, path=None):
    """Coherence vs the number of topics actually found (labels show the nr_topics setting)."""
    s = sweep.sort_values("n_topics_found")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m, c in (("c_v", "#2a78d6"), ("c_npmi", "#eb6834")):
        if m in s:
            ax.plot(s["n_topics_found"], s[m], marker="o", color=c, label=m)
    for setting, r in s.iterrows():
        ax.annotate(
            setting,
            (r["n_topics_found"], r["c_v"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#52514e",
        )
    ax.set_xlabel("number of topics (excluding outliers)")
    ax.set_ylabel("coherence")
    ax.set_title("BERTopic coherence vs number of topics")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# ---------- Guided BERTopic and mapping topics to classes ----------


def seed_list():
    return [words for words in SEED_TOPICS.values()]


def map_topics_by_seeds(model, embeddings=None, topics=None):
    """Unsupervised mapping topic -> class.

    The class whose seed words carry the most c-TF-IDF weight in the topic. If a topic
    contains none of the seed words, use the class whose seed-word centroid
    (embedded with the same encoder) is closest to the topic's mean document embedding."""
    vocab = model.vectorizer_model.get_feature_names_out()
    index = {w: i for i, w in enumerate(vocab)}
    ctfidf = model.c_tf_idf_
    classes = list(SEED_TOPICS)
    seed_emb = None
    mapping = {}
    for t in sorted(model.get_topics()):
        if t == -1:
            continue
        row = ctfidf[t + getattr(model, "_outliers", 1)].toarray().ravel()
        score = np.array(
            [sum(row[index[w]] for w in words if w in index) for words in SEED_TOPICS.values()]
        )
        if score.max() > 0:
            mapping[t] = classes[int(score.argmax())]
            continue
        if seed_emb is None:
            seed_emb = embed_docs([" ".join(w) for w in SEED_TOPICS.values()])
        centroid = embeddings[topics == t].mean(0)
        mapping[t] = classes[int((seed_emb @ centroid).argmax())]
    return mapping


def map_topics_by_majority(topics, labels):
    """Mapping that peeks at the gold labels (each topic -> its most common class). An upper bound,
    not a fair unsupervised result."""
    df = pd.DataFrame({"topic": topics, "label": labels})
    return {t: LABEL_NAMES[g["label"].mode().iloc[0]] for t, g in df.groupby("topic") if t != -1}


def _assign_outliers(model, docs, topics, embeddings):
    """Give every outlier document its nearest topic, so every article gets a class."""
    if (topics == -1).any() and len(set(topics)) > 1:
        topics = np.asarray(
            model.reduce_outliers(docs, list(topics), strategy="embeddings", embeddings=embeddings)
        )
    return topics


def classify_with_topics(model, docs, topics, embeddings, mapping):
    topics = _assign_outliers(model, docs, topics, embeddings)
    fallback = pd.Series(list(mapping.values())).mode().iloc[0] if mapping else "World"
    return np.array([mapping.get(t, fallback) for t in topics]), topics


def guided_topic_table(model, mapping, n_words=8):
    """Topic table with the class each topic was mapped to."""
    table = topic_table(model, n_words)
    table.insert(0, "class", [mapping.get(t, "?") for t in table.index])
    return table


def evaluate_on_ag_news(n=N_AG_EVAL, min_topic_size=10, seed=config.SEED):
    """Fit plain and guided BERTopic on a balanced AG News test sample.

    Map topics to the four classes,
    and score against the gold labels. Returns (scores table, details dict)."""
    from assignment2.data import balanced_subsample, load_ag_news

    _, test = load_ag_news()
    sample = balanced_subsample(test, n, seed=seed)
    docs = sample["text"].tolist()
    emb = embed_docs(docs)
    y = sample["label"].map(LABEL_NAMES).to_numpy()
    rows, details = [], {"sample": sample, "docs": docs, "embeddings": emb}
    for name, seeds in (("BERTopic", None), ("Guided BERTopic", seed_list())):
        t0 = time.perf_counter()
        model, topics = fit_topics(docs, emb, None, seeds, min_topic_size)
        fit_s = time.perf_counter() - t0
        n_topics = len(set(topics) - {-1})
        for how, mapping in (
            ("seed words", map_topics_by_seeds(model, emb, topics)),
            ("majority label (oracle)", map_topics_by_majority(topics, sample["label"])),
        ):
            pred, _ = classify_with_topics(model, docs, topics, emb, mapping)
            rows.append(
                {
                    "method": f"{name}, mapped by {how}",
                    "accuracy": round(accuracy_score(y, pred), 4),
                    "macro_F1": round(f1_score(y, pred, average="macro"), 4),
                    "labels_used": "none" if how == "seed words" else f"{n} test labels",
                    "eval_set": f"AG test sample ({n:,})",
                    "n_topics": n_topics,
                    "fit_seconds": round(fit_s, 1),
                }
            )
            details[(name, how)] = {"model": model, "mapping": mapping, "pred": pred}
    return pd.DataFrame(rows).set_index("method"), details


def comparison_table(unsup_scores):
    """BERTopic scores next to the best supervised classifiers from Assignment 2."""
    sup = pd.DataFrame(A2_SUPERVISED).set_index("method")
    return pd.concat([sup, unsup_scores[["accuracy", "macro_F1", "labels_used", "eval_set"]]])
