"""Load IMDB and draw a random, class-balanced sample."""
import numpy as np
from datasets import load_dataset

from .config import SEED, N_TRAIN, N_TEST


def _balanced_sample(split, n, seed=SEED):
    """Pick n/2 random rows per label (IMDB is sorted by label, so never take the first N)."""
    rng = np.random.default_rng(seed)
    labels = np.array(split["label"])
    idx = np.concatenate([
        rng.choice(np.where(labels == lab)[0], size=n // 2, replace=False)
        for lab in (0, 1)
    ])
    rng.shuffle(idx)
    sub = split.select(idx.tolist())
    return list(sub["text"]), np.array(sub["label"]), idx


def load_imdb_sample(n_train=N_TRAIN, n_test=N_TEST, seed=SEED, return_indices=False):
    ds = load_dataset("stanfordnlp/imdb")
    assert ds["train"].features["label"].names == ["neg", "pos"]   # 0 = negative, 1 = positive
    X_train, y_train, idx_train = _balanced_sample(ds["train"], n_train, seed)
    X_test, y_test, idx_test = _balanced_sample(ds["test"], n_test, seed)
    if return_indices:
        return X_train, y_train, X_test, y_test, {"train": idx_train, "test": idx_test}
    return X_train, y_train, X_test, y_test
