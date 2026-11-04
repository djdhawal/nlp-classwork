"""Shared constants for the experiment."""

SEED = 42
N_TRAIN = 1000          # balanced: 500 pos / 500 neg
N_TEST = 500            # balanced: 250 pos / 250 neg
MAX_SEQ_LENGTH = 256
TOP_K = 10

LABEL_NAMES = {0: "negative", 1: "positive"}

# Embedding models. `query_prefix` comes from each model card.
EMBEDDING_MODELS = {
    "B": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",   # symmetric model, no instruction needed
    },
    "C": {
        "name": "BAAI/bge-small-en-v1.5",
        # bge v1.5 card: add this instruction to short queries for retrieval; passages get no prefix
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}

QUERIES = [
    ("a heartwarming film I would happily watch again", 1),
    ("brilliant performances and a deeply moving story", 1),
    ("one of the best movies I have ever seen", 1),
    ("funny, charming and highly recommended", 1),
    ("terrible acting and a boring plot", 0),
    ("a complete waste of time, I wanted my money back", 0),
    ("poorly written and badly directed", 0),
    ("dull, predictable and a huge disappointment", 0),
    ("not bad at all, much better than I expected", 1),
    ("I expected a great film but it fell flat", 0),
]
