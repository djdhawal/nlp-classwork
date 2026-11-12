# NLP Classwork

Notebooks, assignments, and small apps from an NLP course. It covers text basics, embeddings, transformers, LLM training and fine-tuning, and retrieval-augmented generation (RAG).

## Contents

| Folder / file | What it covers |
|---|---|
| [Assignment-1.ipynb](Assignment-1.ipynb) | Regex, tokenization, text preprocessing, and a TF-IDF news search |
| [Assignment-2.ipynb](Assignment-2.ipynb) | Text classification on AG News: TF-IDF, Word2Vec, BiLSTM, sentence embeddings, and vector search |
| [Exercise-2.ipynb](Exercise-2.ipynb) | TF-IDF vs. sentence embeddings on IMDB, for classification and retrieval |
| [Week3/](Week3/) | Self-attention, positional encodings, BERT (MLM and QA), SBERT and ColBERT, Matryoshka embeddings, contrastive fine-tuning, MTEB |
| [Week4/](Week4/) | LLM training data, deduplication, parallelism, mixture of experts, multi-head latent attention, model families, and BabyQwen pretraining |
| [Week5/](Week5/) | LoRA and PEFT, RLHF / DPO / GRPO, decoding strategies, LLM evaluation, and BabyQwen fine-tuning and knowledge injection |
| [Week6/](Week6/) | RAG end to end: ingestion (web, PDF, OCR, speech), chunking, vector databases, hybrid retrieval, query rewriting, reranking, prompting, evaluation, and citations |
| [Week7/](Week7/) | Exercise 7: a local, always-on voice assistant (wake word, speech to text, tool-calling agent, memory, text to speech, Streamlit page) |

HTML exports of the graded notebooks sit next to the `.ipynb` files.

## Code packages and apps

| Path | What it is |
|---|---|
| [assignment2/](assignment2/) | Code for Assignment 2: data loading, EDA, classic models, embeddings, Qdrant vector DB, and a Streamlit search app |
| [exercise2/](exercise2/) | Code for Exercise 2: features, classification, and retrieval |
| [news_search_app.py](news_search_app.py) | Streamlit TF-IDF news search (Assignment 1) |
| [news_search/](news_search/) | Qdrant + SentenceTransformers news search. See its [Readme](news_search/Readme.md) |
| [Week6/news-scrapper/](Week6/news-scrapper/) | Scrapy spider for financial news sites. See its [README](Week6/news-scrapper/README.md) |
| [Week6/rag_common.py](Week6/rag_common.py) | Shared helpers for the Week 6 RAG notebooks |
| [Week7/voice-assistant/](Week7/voice-assistant/) | Exercise 7 voice assistant "Loop". See its [README](Week7/voice-assistant/README.md) |

## Setup

The notebooks were run in a conda env named `ml_hw` (Python 3.11). There is no single requirements file. The main packages are:

```bash
pip install numpy pandas scikit-learn matplotlib torch transformers sentence-transformers \
            datasets qdrant-client streamlit jupyter
```

Some notebooks need more (for example `scrapy`, `peft`, `trl`, OCR or speech libraries). Install them when a notebook asks.

### Qdrant

Assignment 2 uses embedded Qdrant, so no server is needed. The Week 6 notebooks and `news_search/` need a Qdrant server:

```bash
cd Week6
docker compose up -d     # Qdrant at http://localhost:6333
```

## Running the apps

Run from the repo root.

```bash
# Assignment 1: TF-IDF news search
streamlit run news_search_app.py

# Assignment 2: vector search (build the Qdrant collection in the notebook first)
streamlit run assignment2/search_app.py
```

## Data

Large data files (`*.csv`, `*.json`, model weights, caches) are git-ignored. The notebooks download what they need, such as AG News from Kaggle and IMDB from Hugging Face. Small sample files for Week 6 are in [Week6/data/](Week6/data/).
