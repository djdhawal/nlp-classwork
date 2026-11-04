"""Load and clean AG News (Kaggle) and the scraped news articles from Assignment 1."""
import html
import re

import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import KAGGLE_DATASET, SCRAPED_CSV, SCRAPED_LABELS, SEED

STOPWORDS = ENGLISH_STOP_WORDS | {"said", "says", "new", "reuters", "ap", "afp", "quot", "lt", "gt",
                                      "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


def light_clean(text):
    """Fix AG News artifacts (HTML entities, '#36;' codes, backslashes, 'Reuters - ' bylines).
    Keeps case and stop words, so it is the right input for sentence encoders."""
    text = html.unescape(str(text))
    text = re.sub(r"#\d+;", " ", text)                      # e.g. '#36;' for '$'
    text = re.sub(r"\bamp;", "&", text).replace("quot;", '"')  # entities with the '&' already stripped
    text = re.sub(r"&lt;.*?&gt;|<[^>]+>", " ", text)          # leftover tags
    text = text.replace("\\", " ")
    text = re.sub(r"^\s*[A-Z][\w.,' ]{0,40}\s\((Reuters|AP|AFP)\)\s*-+\s*", "", text)  # 'LONDON (Reuters) - '
    text = re.sub(r"^\s*(Reuters|AP|AFP|[A-Z][\w.]+\.com)\s*-+\s*", "", text)          # 'Reuters - '
    return re.sub(r"\s+", " ", text).strip()


def heavy_clean(text):
    """Lowercase, drop special characters, numbers and stop words. Input for TF-IDF / Word2Vec."""
    tokens = re.findall(r"[a-z]+", light_clean(text).lower())
    return " ".join(t for t in tokens if t not in STOPWORDS and len(t) > 1)


def load_ag_news():
    """Download AG News from Kaggle (cached by kagglehub) and add cleaned text columns."""
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    out = []
    for split in ("train", "test"):
        df = pd.read_csv(f"{path}/{split}.csv").rename(
            columns={"Class Index": "label", "Title": "title", "Description": "description"})
        df = df.dropna().drop_duplicates(subset=["title", "description"]).reset_index(drop=True)
        df["text"] = (df["title"].map(light_clean) + ". " + df["description"].map(light_clean))
        df["clean"] = df["text"].map(heavy_clean)
        out.append(df)
    return out[0], out[1]


def balanced_subsample(df, n, seed=SEED):
    """n/4 random rows per class."""
    per = n // df["label"].nunique()
    return (df.groupby("label").sample(per, random_state=seed)
              .sample(frac=1, random_state=seed).reset_index(drop=True))


def _lede(body, max_chars=300):
    """First sentence(s) of an article, so scraped text looks like an AG News description."""
    body = re.sub(r"^Updated on:.*?(EDT|EST)\s+CBS( News| AP)?\s*", "", str(body))
    body = re.sub(r"^Your support helps make our show possible.*?feed\.\s*", "", body)
    cut = body[:max_chars]
    end = cut.rfind(". ")
    return cut[: end + 1] if end > 80 else cut


def load_scraped():
    """Scraped articles with the hand labels from scraped_labels.csv (same 1-4 scheme)."""
    df = pd.read_csv(SCRAPED_CSV).fillna("")
    labels = pd.read_csv(SCRAPED_LABELS)
    assert (labels["title"].values == df["title"].values).all(), "labels out of sync with news_clean.csv"
    df["label"] = labels["label"].values
    df["text"] = (df["title"].map(light_clean) + ". " + df["text_clean"].map(_lede).map(light_clean))
    df["clean"] = df["text"].map(heavy_clean)
    return df
