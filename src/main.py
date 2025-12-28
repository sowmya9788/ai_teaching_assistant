import os
import sys
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.document_processor import DocumentProcessor
from src.embedding.embeddings_manager import EmbeddingsManager
from src.qa.rag_pipeline import RAGPipeline
from src.config import Config

class TeachingAssistant:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embeddings_manager = EmbeddingsManager()
        self.rag_pipeline = RAGPipeline()
        self.documents_processed = 0
    
    def upload_and_process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            # Process document
            chunks = self.processor.process_document(file_path)
            
            # Chunk text for better retrieval
            chunked_docs = self.processor.chunk_text(chunks, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
            
            # Generate embeddings
            embedded_chunks = self.embeddings_manager.embed_chunks(chunked_docs)
            
            # Add to RAG pipeline
            self.rag_pipeline.add_document(embedded_chunks)
            
            self.documents_processed += 1
            
            return {
                "status": "success",
                "message": f"Document processed successfully. Created {len(chunked_docs)} chunks.",
                "chunks_created": len(chunked_docs),
                "total_documents": self.documents_processed
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}"
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        if self.documents_processed == 0:
            return {
                "status": "error",
                "message": "No documents have been uploaded yet. Please upload lecture notes first.",
                "answer": "",
                "reference": ""
            }
        
        try:
            result = self.rag_pipeline.answer_question(question)
            
            # Format the response
            answer = result["answer"]
            sources = result["sources"]
            
            # Get the best reference
            reference = ""
            if sources:
                best_source = sources[0]
                ref_type = best_source.get("reference_type", "Page")
                page_num = best_source.get("page", "Unknown")
                reference = f"{ref_type} {page_num}"
            
            return {
                "status": "success",
                "answer": answer,
                "reference": reference,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error answering question: {str(e)}",
                "answer": "",
                "reference": ""
            }
    
    def get_document_count(self) -> int:
        return self.rag_pipeline.get_document_count()
    
    def clear_all_documents(self):
        self.rag_pipeline.clear_documents()
        self.documents_processed = 0

# Global instance
teaching_assistant = TeachingAssistant()
