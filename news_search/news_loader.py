import pandas as pd
import json
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

def load_vectors(data, client, collection_name):
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(data.head())
    for index, article in enumerate(data.itertuples()):
        # Read the dataset
        article_id = index
        title = article.Title
        category = article.Category
        text = article.Description

        # Convert text to vector
        vector = model.encode(title, convert_to_tensor=True).tolist()

        # Insert feature vector and metadata into Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=article_id,
                    payload= {"id":article_id, "category": category, "text": text,"title": title},
                    vector=vector,
                ),
            ],
        )
        print("Inserting : ", article_id)

def main():    
    collection_name = "news"
    DATA_FILE_PATH = 'news.csv'

    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")

    # Create the collection in Qdrant
    if not client.collection_exists(collection_name):
        print(f"Creating Collection {collection_name}.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    #Load Data
    data = pd.read_csv(DATA_FILE_PATH)
    data.columns = ['Category', 'Title', 'Description']

    load_vectors(data, client, collection_name)

if __name__ == "__main__":
    main()
