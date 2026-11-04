"""Section 2.4: cosine-similarity heatmaps and checks that similar articles score higher."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pairs of scraped articles that cover the same story from different outlets (indices into news_clean.csv).
SAME_STORY_PAIRS = [(0, 43), (1, 38), (2, 19), (6, 30), (8, 39), (17, 24), (20, 34), (42, 49)]


def short(title, n=45):
    return title if len(title) <= n else title[: n - 1] + "…"


def heatmaps(embs, titles, order=None):
    """One cosine-similarity heatmap per model, same article order for all."""
    order = list(range(len(titles))) if order is None else list(order)
    labels = [short(titles[i]) for i in order]
    ncols = min(2, len(embs))
    nrows = -(-len(embs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6.5 * nrows), squeeze=False)
    for ax in axes.flat[len(embs):]:
        ax.axis("off")
    for ax, (name, E) in zip(axes.flat, embs.items()):
        S = E[order] @ E[order].T
        im = ax.imshow(S, cmap="viridis", vmin=np.percentile(S, 5), vmax=1)
        ax.set_title(name)
        ax.set_xticks(range(len(order)), range(1, len(order) + 1), fontsize=7)
        ax.set_yticks(range(len(order)), [f"{i+1}. {l}" for i, l in enumerate(labels)], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout()


def pair_scores(E, pairs=SAME_STORY_PAIRS):
    """For each same-story pair: its cosine, the mean cosine of each article to all others,
    and the rank of the partner among the article's neighbours (1 = nearest)."""
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    rows = []
    for a, b in pairs:
        others = np.delete(S[a], [a, b])
        rows.append({"pair": (a, b), "cos": S[a, b], "mean_cos_to_others": others.mean(),
                     "partner_rank": int((S[a] > S[a, b]).sum()) + 1})
    return pd.DataFrame(rows)


def nearest_neighbours(E, titles, idx, k=3):
    S = E @ E[idx]
    S[idx] = -np.inf
    top = np.argsort(-S)[:k]
    return pd.DataFrame({"neighbour": [titles[i] for i in top], "cos": S[top].round(3)})
