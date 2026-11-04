"""Feature builders: TF-IDF (model A) and sentence embeddings (models B, C).

Each builder returns train/test features, a function to encode queries, and build time.
"""
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .config import MAX_SEQ_LENGTH


def build_tfidf(X_train, X_test, **tfidf_kwargs):
    t0 = time.perf_counter()
    vec = TfidfVectorizer(**{"sublinear_tf": True, "min_df": 2, **tfidf_kwargs})
    F_train = vec.fit_transform(X_train)        # fit on training split only
    F_test = vec.transform(X_test)
    elapsed = time.perf_counter() - t0
    return {"train": F_train, "test": F_test, "encode_queries": vec.transform, "seconds": elapsed,
            "vectorizer": vec}


def build_embeddings(model_name, X_train, X_test, query_prefix="", batch_size=32):
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    enc = lambda texts: model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=False)
    F_train = enc(X_train)
    F_test = enc(X_test)
    elapsed = time.perf_counter() - t0
    encode_queries = lambda qs: enc([query_prefix + q for q in qs])
    return {"train": F_train, "test": F_test, "encode_queries": encode_queries,
            "seconds": elapsed, "model": model}
