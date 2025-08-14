import os

class Config:
    """Flask configuration variables."""
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a-very-secret-key')

    # LLM Config
    DEFAULT_MODEL = "gpt-oss:20b" # Or any other default model you prefer
    PERSIST_DIRECTORY = os.path.join(os.path.abspath(os.path.dirname(__file__)), "conversation_db")
    EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # --- Tuning Parameters ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retriever
    VECTOR_SEARCH_K = 5  # k for vector search
    BM25_SEARCH_K = 5    # k for BM25 search
    ENSEMBLE_WEIGHTS = [0.5, 0.5] # Weights for BM25 and Vector search

    # Embedding Device (cpu, cuda, mps etc.)
    EMBEDDING_DEVICE = "cpu"

    # LLM Request Timeout
    OLLAMA_REQUEST_TIMEOUT = 600.0 # Timeout for requests to Ollama in seconds (e.g., 600.0 = 10 minutes)