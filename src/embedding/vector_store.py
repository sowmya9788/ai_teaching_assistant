import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from src.config import Config

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.vector_db_type = Config.VECTOR_DB_TYPE
        self.embedding_dim = None
        
        if self.vector_db_type == "faiss" and faiss is None:
            # faiss-cpu is not available on all Python versions/platforms (notably Python 3.13).
            # Fall back to Chroma to keep the app working end-to-end.
            self.vector_db_type = "chroma"

        if self.vector_db_type == "chroma":
            self._init_chroma()
        else:
            self._init_faiss()
    
    def _init_faiss(self):
        self.index = None
        self.chunks = []
        self.vector_db_type = "faiss"
    
    def _init_chroma(self):
        os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(
            name="lecture_notes",
            metadata={"hnsw:space": "cosine"}
        )
        self.vector_db_type = "chroma"
    
    def add_embeddings(self, embedded_chunks: List[Dict[str, Any]]):
        if self.vector_db_type == "chroma":
            self._add_chroma(embedded_chunks)
        else:
            self._add_faiss(embedded_chunks)
    
    def _add_faiss(self, embedded_chunks: List[Dict[str, Any]]):
        if faiss is None:
            raise RuntimeError("FAISS is not available in this environment. Use VECTOR_DB_TYPE=chroma instead.")
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
        
        if self.index is None:
            self.embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.index.add(embeddings)
        self.chunks.extend(embedded_chunks)
    
    def _add_chroma(self, embedded_chunks: List[Dict[str, Any]]):
        ids = [chunk['chunk_id'] for chunk in embedded_chunks]
        embeddings = [chunk['embedding'].tolist() for chunk in embedded_chunks]
        documents = [chunk['text'] for chunk in embedded_chunks]
        metadatas = [chunk['metadata'] for chunk in embedded_chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        if self.vector_db_type == "chroma":
            return self._search_chroma(query_embedding, top_k)
        else:
            return self._search_faiss(query_embedding, top_k)
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if faiss is None:
            return []
        if self.index is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append(chunk)
        
        return results
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': results['distances'][0][i] if 'distances' in results else 1.0
            }
            formatted_results.append(chunk)
        
        return formatted_results
    
    def save_index(self, file_path: str):
        if self.vector_db_type == "faiss" and faiss is not None:
            faiss.write_index(self.index, file_path)
            with open(file_path + "_chunks.pkl", 'wb') as f:
                pickle.dump(self.chunks, f)
    
    def load_index(self, file_path: str):
        if self.vector_db_type == "faiss" and faiss is not None and os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
            with open(file_path + "_chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
    
    def clear(self):
        if self.vector_db_type == "chroma":
            self.client.delete_collection("lecture_notes")
            self.collection = self.client.get_or_create_collection(
                name="lecture_notes",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.index = None
            self.chunks = []
