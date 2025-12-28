import numpy as np
from typing import List, Dict, Any
import openai
from src.ingestion.document_processor import DocumentChunk
from src.config import Config

class EmbeddingsManager:
    def __init__(self):
        self.model = None
        self.embedding_dim = None
        self._initialize_model()
    
    def _initialize_model(self):
        if Config.OPENAI_API_KEY:
            try:
                # Try OpenAI embeddings first
                openai.api_key = Config.OPENAI_API_KEY
                self.model_type = "openai"
                self.embedding_dim = 1536  # OpenAI ada-002 dimension
            except:
                print("Failed to initialize OpenAI, falling back to HuggingFace")
                self._init_huggingface()
        else:
            self._init_huggingface()
    
    def _init_huggingface(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(Config.HF_MODEL_NAME)
            self.model_type = "huggingface"
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback for environments where sentence-transformers/torch are not available.
            # Uses a stateless vectorizer so we can embed both documents and queries consistently.
            from sklearn.feature_extraction.text import HashingVectorizer

            self.vectorizer = HashingVectorizer(
                n_features=1024,
                alternate_sign=False,
                norm="l2",
                stop_words="english",
            )
            self.model_type = "sklearn_hashing"
            self.embedding_dim = 1024
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model_type == "openai":
            return self._get_openai_embeddings(texts)
        elif self.model_type == "huggingface":
            return self._get_huggingface_embeddings(texts)
        else:
            return self._get_sklearn_hashing_embeddings(texts)
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _get_huggingface_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def _get_sklearn_hashing_embeddings(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return X.toarray().astype(np.float32)
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = {
                'text': chunk.text,
                'embedding': embeddings[i],
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
