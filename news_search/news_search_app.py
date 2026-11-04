import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import time

# Set wide mode
st.set_page_config(layout="wide")

# Streamlit application setup
st.title("News Search")

# Sidebar Configuration for Server and Model
st.sidebar.header("Configuration")

# Server URL options
server_options = [
    "http://localhost:6333",
    "http://127.0.0.1:6333",
    "http://your-server-ip:6333",  # Example, adjust with your actual server IP
]
server_url = st.sidebar.selectbox("Select Qdrant Server URL", server_options, index=0)

# Full server URL
qdrant_host = f"{server_url}"

# Collection Name input
collection_name = st.sidebar.text_input("Collection Name", "news")

# Add results limit slider
results_limit = st.sidebar.number_input("Number of results", min_value=5, max_value=100, value=10, step=5)

# Model selection dropdown
model_options = {
    "all-MiniLM-L6-v2": "Sentence-Transformer: all-MiniLM-L6-v2"
    # Add more models here
}
selected_model = st.sidebar.selectbox("Select Embedding Model", list(model_options.keys()))

# Load selected model
model = SentenceTransformer(selected_model)

# Add category filter
categories = ["All", "1", "2", "3", "4"]  # Add your categories here
selected_category = st.sidebar.selectbox("Filter by Category", categories)

# User input for search term
query = st.text_input("Enter search term:")

def search_news(query, qdrant_host, collection_name, limit, category=None):
    # Convert the query text to a vector using the selected model
    vector = model.encode(query, convert_to_tensor=True).tolist()

    # Initialize Qdrant client
    qdrant_client = QdrantClient(qdrant_host)

    # Create filter if category is selected
    search_filter = None
    if category and category != "All":
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=int(category))
                )
            ]
        )

    # Perform the search with optional filter
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=vector,  
        query_filter=search_filter,  
        limit=limit
    )
    return results

if st.button("Search"):
    if query:
        # Start timing
        start_time = time.time()
        
        # Pass the category filter to the search function
        results = search_news(query, qdrant_host, collection_name, results_limit, selected_category)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        st.info(f"Search completed in {execution_time:.2f} seconds")

        # Process results into a DataFrame for nicer display
        if results:
            data = [{
                "Article ID": result.id,
                "Title": result.payload['title'],
                "Text": result.payload['text'],
                "Score": result.score
            } for result in results.points]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a search term to proceed.")
