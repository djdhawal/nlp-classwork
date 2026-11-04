"""Shared constants: paths, labels, sample sizes and model lists."""
from pathlib import Path

SEED = 42
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
CACHE_DIR = PKG_DIR / "cache"            # cached embeddings (.npy)
QDRANT_PATH = PKG_DIR / "qdrant_db"      # embedded Qdrant storage (no server needed)
SCRAPED_CSV = ROOT / "news_clean.csv"    # articles scraped in Assignment 1
SCRAPED_LABELS = PKG_DIR / "scraped_labels.csv"

KAGGLE_DATASET = "amananandrai/ag-news-classification-dataset"
LABEL_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# Embedding models are slower than TF-IDF, so they use a balanced subsample of the train split.
N_EMB_TRAIN = 20_000                     # 5,000 per class
BATCH_SIZE = 128

# Section 2.3: encoders picked for classification (small enough for a laptop).
# `prefix` is what each model card asks for on classification inputs.
CLASSIFICATION_MODELS = {
    "all-MiniLM-L6-v2": {"name": "sentence-transformers/all-MiniLM-L6-v2", "prefix": ""},
    "bge-small-en-v1.5": {"name": "BAAI/bge-small-en-v1.5", "prefix": ""},
    "gte-small": {"name": "thenlper/gte-small", "prefix": ""},
    "e5-small-v2": {"name": "intfloat/e5-small-v2", "prefix": "query: "},
    "bge-base-en-v1.5": {"name": "BAAI/bge-base-en-v1.5", "prefix": ""},
}

# Sections 2.4 and 2.5: encoders picked for search. Asymmetric models use a query prefix,
# and e5 also wants a passage prefix on documents.
SEARCH_MODELS = {
    "all-MiniLM-L6-v2": {"name": "sentence-transformers/all-MiniLM-L6-v2",
                         "query_prefix": "", "doc_prefix": ""},
    "multi-qa-MiniLM-L6": {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                           "query_prefix": "", "doc_prefix": ""},
    "bge-small-en-v1.5": {"name": "BAAI/bge-small-en-v1.5",
                          "query_prefix": "Represent this sentence for searching relevant passages: ",
                          "doc_prefix": ""},
    "e5-small-v2": {"name": "intfloat/e5-small-v2",
                    "query_prefix": "query: ", "doc_prefix": "passage: "},
}
SEARCH_MODEL = "bge-small-en-v1.5"       # model behind the vector DB and the Streamlit app
COLLECTION = "ag_news"
