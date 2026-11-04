You are an expert Python developer specializing in LLM applications.

Build a complete Streamlit app that allows users to experiment with advanced Retrieval-Augmented Generation (RAG) strategies. The app should be modular, well-structured, and easy to extend.

### Core Requirements

1. **User Interface (Streamlit)**

   - Form field to upload a PDF file
   - Text input box for user query
   - Dropdowns for:
     - Chunking Strategy:
       - Fixed-size chunking
       - Recursive chunking
       - Semantic chunking
     - Embedding Model (via Ollama, local):
       - nomic-embed-text
       - mxbai-embed-large
     - LLM Model (via Ollama, local):
       - gemma4:latest
       - mistral
     - Retrieval Strategy:
       - Standard similarity search
       - HyDE (Hypothetical Document Embeddings)
       - FLARE (Forward-Looking Active Retrieval)
   - Slider for:
     - Chunk size
     - Top-K retrieval
   - Button to run query

3. **Backend Logic**
   - Use Qdrant as the local vector database
   - Connect to Qdrant running locally (assume default localhost:6333)
   - Store and retrieve embeddings from Qdrant collections

4. **Chunking Implementations**
   - Implement:
     - Fixed chunking (simple splits)
     - Recursive text splitting (like LangChain RecursiveCharacterTextSplitter)
     - Semantic chunking (use embeddings similarity if possible)

5. **Embedding + Storage**
   - Use Ollama embeddings via API
   - Ensure embeddings are stored in Qdrant
   - Include a simple document loader (text file upload or predefined text)

6. **Retrieval Strategies**
   - **Standard RAG**:
     - Embed query → similarity search → pass context to LLM
   - **HyDE**:
     - Generate hypothetical answer using LLM
     - Embed that answer → retrieve documents
   - **FLARE (simplified)**:
     - Iteratively retrieve more context based on partial generation

7. **Response Generation**
   - Use selected Ollama LLM to generate answers
   - Show:
     - Final answer
     - Retrieved chunks (expandable section)
     - Debug info (strategy used, time taken)

8. **Code Quality**
   - Use clean modular functions:
     - chunk_text()
     - embed_text()
     - retrieve_docs()
     - run_rag_pipeline()
   - Add comments explaining each RAG strategy
   - Use environment variables where needed

9. **Bonus Features (if possible)**
   - Cache embeddings
   - Display similarity scores
   - Allow switching Qdrant collections

### Constraints
- Use only Python
- Use Streamlit for UI
- Use Ollama for local models (no OpenAI API)
- Use Qdrant Python client
- Keep everything runnable in a single script (or minimal files)

### Output
- Provide full working code
- Include installation instructions
- Include example commands to run Ollama models