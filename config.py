import os

class Config:
    """Flask configuration variables."""
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a-very-secret-key')

    # RAG Config
    DEFAULT_MODEL = "deepseek-r1:8b" # Or any other default model you prefer
    PERSIST_DIRECTORY = os.path.join(os.path.abspath(os.path.dirname(__file__)), "conversation_db")
    EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    OLLAMA_BASE_URL = "http://localhost:11434"
