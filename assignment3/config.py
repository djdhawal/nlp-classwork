"""Shared settings: paths, models and RAG parameters. Every module reads from here."""
import os
import sys
from pathlib import Path

SEED = 42
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

# Data (committed, so a Colab clone has it) and outputs (gitignored; on Colab they go to Drive).
DATA_DIR = PKG_DIR / "data"
ARTICLES_CSV = DATA_DIR / "news_articles.csv"      # columns: article_id, source, url, title, date, text
OUT_DIR = Path(os.environ.get("A3_OUT_DIR", PKG_DIR / "outputs"))
CACHE_DIR = OUT_DIR / "cache"                        # embeddings (.npy), LLM reply cache
QDRANT_PATH = OUT_DIR / "qdrant_db"                  # embedded Qdrant, no server needed
RESULTS_DIR = OUT_DIR / "results"                    # tables and figures for the report

# Scraping
N_ARTICLES = 500

# Chunking (baseline)
CHUNK_SIZE = 1000            # characters
CHUNK_OVERLAP = 150

# Embeddings: the search model picked in Assignment 2 (bge-small needs a query prefix).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC_PREFIX = ""
BATCH_SIZE = 64
COLLECTION = "news_chunks"
TOP_K = 4

# Re-ranker used by the improved pipeline in 3.3 B.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLMs. Everything goes through an OpenAI-compatible API, so Ollama (local or Colab),
# OpenAI and OpenRouter are interchangeable. A model is "provider:name".
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PROVIDERS = {
    "ollama": {"base_url": OLLAMA_URL + "/v1", "key_env": None},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
}
LLM = os.environ.get("A3_LLM", "ollama:qwen2.5:7b-instruct")          # default generator
COMPARE_LLMS = ["ollama:qwen2.5:7b-instruct", "ollama:llama3.2:3b"]   # 3.2 step 6
JUDGE_LLM = os.environ.get("A3_JUDGE", "ollama:qwen2.5:7b-instruct")  # DeepEval judge
TEMPERATURE = 0.0
MAX_TOKENS = 512

# Guardrails
MIN_RETRIEVAL_SCORE = 0.55   # cosine similarity; below this for every chunk -> abstain
MAX_QUERY_CHARS = 500


def ollama_models():
    """Ollama models the notebook needs pulled (generators + judge)."""
    names = [LLM, *COMPARE_LLMS, JUDGE_LLM]
    return list(dict.fromkeys(n.split(":", 1)[1] for n in names if n.startswith("ollama:")))


def ensure_dirs():
    for d in (DATA_DIR, OUT_DIR, CACHE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
