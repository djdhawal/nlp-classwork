"""Sanity checks for the experiment setup. Each check raises AssertionError if it fails."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import N_TRAIN, N_TEST, QUERIES, LABEL_NAMES

# The 10 queries copied verbatim from the assignment, with the sentiment each one wants.
EXPECTED_QUERIES = [
    ("a heartwarming film I would happily watch again", "positive"),
    ("brilliant performances and a deeply moving story", "positive"),
    ("one of the best movies I have ever seen", "positive"),
    ("funny, charming and highly recommended", "positive"),
    ("terrible acting and a boring plot", "negative"),
    ("a complete waste of time, I wanted my money back", "negative"),
    ("poorly written and badly directed", "negative"),
    ("dull, predictable and a huge disappointment", "negative"),
    ("not bad at all, much better than I expected", "positive"),
    ("I expected a great film but it fell flat", "negative"),
]


def check_sample(y_train, y_test, indices):
    for name, y, n, idx in (("train", y_train, N_TRAIN, indices["train"]),
                            ("test", y_test, N_TEST, indices["test"])):
        counts = np.bincount(y)
        print(f"{name}: n={len(y)}  neg={counts[0]}  pos={counts[1]}  "
              f"row indices span {idx.min()}..{idx.max()}")
        assert len(y) == n and counts[0] == counts[1] == n // 2, "sample is not balanced"
        assert not np.array_equal(np.sort(idx), np.arange(n)), "took the first N rows"
        assert len(set(idx)) == n, "duplicate rows sampled"
    print("OK: random, balanced, not the first N rows")


def check_tfidf_no_leakage(tfidf, X_train, X_test):
    vocab = set(tfidf["vectorizer"].vocabulary_)
    ref = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(X_train)
    assert vocab == set(ref.vocabulary_), "TF-IDF vocabulary differs from a train-only fit"
    test_only = set(TfidfVectorizer(min_df=2).fit(X_test).vocabulary_) - set(
        TfidfVectorizer(min_df=1).fit(X_train).vocabulary_)
    leaked = test_only & vocab
    print(f"vocab={len(vocab)}  words seen only in test: {len(test_only)}  of those in vocab: {len(leaked)}")
    assert not leaked, "test-only words leaked into the vocabulary"
    print("OK: TF-IDF fit on the training split only")


def check_no_prefix_on_reviews(feat, X_test, n=3):
    """Re-encode a few reviews with no prefix; they must match the stored test vectors."""
    fresh = feat["model"].encode(X_test[:n], normalize_embeddings=True, show_progress_bar=False)
    cos = (fresh * feat["test"][:n]).sum(axis=1)
    assert np.allclose(cos, 1.0, atol=1e-4), f"stored review vectors differ (cos={cos})"
    print(f"OK: reviews encoded without prefix (cos to stored vectors = {np.round(cos, 5)})")


def check_queries():
    got = [(q, LABEL_NAMES[w]) for q, w in QUERIES]
    assert got == EXPECTED_QUERIES, "queries or intents differ from the assignment"
    print("OK: 10 queries verbatim; intents map to labels", {v: k for k, v in LABEL_NAMES.items()})
