"""Section 2.2 EDA: class balance, text lengths, top words and word clouds per class."""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from .config import LABEL_NAMES

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6"]   # one fixed color per class, 1..4


def class_distribution(train, test):
    counts = pd.DataFrame({"train": train["label"].value_counts(), "test": test["label"].value_counts()})
    counts.index = counts.index.map(LABEL_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col in zip(axes, counts):
        ax.bar(counts.index, counts[col], color=COLORS)
        ax.set_title(f"{col} ({counts[col].sum():,} rows)")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return counts


def length_stats(df):
    lens = df.assign(words=df["text"].str.split().str.len(),
                     clean_words=df["clean"].str.split().str.len())
    stats = lens.groupby(lens["label"].map(LABEL_NAMES))[["words", "clean_words"]].describe()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for (lab, name), c in zip(LABEL_NAMES.items(), COLORS):
        ax.hist(lens.loc[lens["label"] == lab, "words"], bins=60, range=(0, 120), alpha=0.5,
                color=c, label=name)
    ax.set_xlabel("words per article (title + description)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return stats.round(1)


def top_words(df, n=15):
    return pd.DataFrame({
        name: [w for w, _ in Counter(" ".join(df.loc[df["label"] == lab, "clean"]).split()).most_common(n)]
        for lab, name in LABEL_NAMES.items()
    })


def word_clouds(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (lab, name) in zip(axes, LABEL_NAMES.items()):
        text = " ".join(df.loc[df["label"] == lab, "clean"])
        wc = WordCloud(width=600, height=400, background_color="white", max_words=100,
                       collocations=False, random_state=0).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(name, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
