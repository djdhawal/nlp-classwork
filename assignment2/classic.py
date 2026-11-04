"""Section 2.2: TF-IDF / Word2Vec features, several ML classifiers, and evaluation."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import LABEL_NAMES, SEED


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Train Word2Vec on the training texts; a document is the mean of its word vectors."""

    def __init__(self, vector_size=200, window=5, min_count=2, epochs=10):
        self.vector_size, self.window, self.min_count, self.epochs = vector_size, window, min_count, epochs

    def fit(self, texts, y=None):
        sents = [t.split() for t in texts]
        self.w2v_ = Word2Vec(sents, vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, epochs=self.epochs, workers=4, seed=SEED)
        return self

    def transform(self, texts):
        wv = self.w2v_.wv
        out = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs = [wv[w] for w in t.split() if w in wv.key_to_index]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)


def make_pipelines():
    """Encoding + classifier pairs. Each pipeline is fit on the cleaned training text only."""
    return {
        "TF-IDF + LogisticRegression": Pipeline([("vec", tfidf()),
                                                 ("clf", LogisticRegression(C=5, max_iter=2000))]),
        "TF-IDF + LinearSVC": Pipeline([("vec", tfidf()), ("clf", LinearSVC(C=0.5))]),
        "TF-IDF + ComplementNB": Pipeline([("vec", tfidf()), ("clf", ComplementNB(alpha=0.3))]),
        "Word2Vec + LogisticRegression": Pipeline([("vec", Word2VecVectorizer()), ("scale", StandardScaler()),
                                                   ("clf", LogisticRegression(max_iter=2000))]),
        "Word2Vec + RandomForest": Pipeline([("vec", Word2VecVectorizer()),
                                             ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                                            n_jobs=-1, random_state=SEED))]),
    }


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_F1": f1_score(y_true, y_pred, average="macro"),
            "weighted_F1": f1_score(y_true, y_pred, average="weighted")}


def train_all(pipelines, X_train, y_train, X_test, y_test):
    rows, fitted = [], {}
    for name, pipe in pipelines.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        rows.append({"model": name, **scores(y_test, pipe.predict(X_test)), "fit_seconds": fit_s})
        fitted[name] = pipe
    return pd.DataFrame(rows).set_index("model").round(4), fitted


def report(y_true, y_pred, title=""):
    """Print per-class metrics and draw the confusion matrix (only for labels that occur)."""
    labels = sorted(set(y_true) | set(y_pred))
    names = [LABEL_NAMES[l] for l in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=names,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
