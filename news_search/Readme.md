# News Search Engine

A vector-based news search engine using Qdrant and SentenceTransformers.

## Prerequisites

- Python 3.7+
- News dataset in CSV format
- Qdrant running locally (default: http://localhost:6333/dashboard)

## Installation

Install docker container

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Install the required packages:

```bash
pip install qdrant-client --upgrade
pip install sentence-transformers
pip install pandas
pip install streamlit
```

## Usage

1. Ensure your news dataset is in CSV format with columns:
   - ClassIndex
   - Title
   - Description

2. Place the dataset as `news.csv` in the project directory

3. Run the loader:

```bash
python news_loader.py
```

This will create a Qdrant collection named "news" and populate it with vector embeddings of the news articles.

4. Run the Streamlit app:

```bash
streamlit run news_search_app.py
```