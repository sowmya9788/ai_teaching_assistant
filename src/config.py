import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector Database
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", 10))
    
    # RAG Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    
    # File Paths
    UPLOAD_DIR = "./uploads"
    DATA_DIR = "./data"
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY and cls.VECTOR_DB_TYPE == "openai":
            raise ValueError("OpenAI API key is required when using OpenAI embeddings")
