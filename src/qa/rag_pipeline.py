import openai
import numpy as np
from typing import List, Dict, Any, Optional
from src.config import Config
from src.embedding.embeddings_manager import EmbeddingsManager
from src.embedding.vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.vector_store = VectorStore()
        self._init_llm()
    
    def _init_llm(self):
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.llm_type = "openai"
        else:
            self.llm_type = "huggingface"
            # For HuggingFace, you could use transformers pipeline
            # For simplicity, we'll use a basic approach
    
    def add_document(self, embedded_chunks: List[Dict[str, Any]]):
        self.vector_store.add_embeddings(embedded_chunks)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        # Generate embedding for the question
        query_embedding = self.embeddings_manager.get_embeddings([question])[0]
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(query_embedding, Config.TOP_K_RETRIEVAL)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in your lecture notes to answer this question.",
                "sources": [],
                "question": question
            }
        
        # Format sources (includes doc_type + reference_type)
        sources = self._format_sources(relevant_chunks)

        # Compute reference strictly from retrieved metadata (no fabrication)
        reference = "Page - / Slide -"
        if sources:
            best = sources[0]
            ref_type = best.get("reference_type", "Page")
            ref_num = best.get("page", "-")
            if ref_type == "Slide":
                reference = f"Page - / Slide {ref_num}"
            else:
                reference = f"Page {ref_num} / Slide -"

        # Generate answer based on retrieved chunks (exam-ready expansion)
        answer = self._generate_answer(question, relevant_chunks, reference)
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }
    
    def _generate_answer(self, question: str, chunks: List[Dict[str, Any]], reference: str) -> str:
        context = self._build_context(chunks)
        excerpts = self._build_excerpts(chunks)
        
        if self.llm_type == "openai":
            return self._generate_openai_answer(question, context, excerpts, reference)
        else:
            return self._generate_simple_answer(question, context, excerpts, reference)

    def _build_excerpts(self, chunks: List[Dict[str, Any]], max_excerpts: int = 3) -> str:
        """Short quoted lines from the retrieved text to show what was used.

        This is separate from the long context to make it easy for the LLM/UI to highlight.
        """
        excerpts = []
        for chunk in chunks[:max_excerpts]:
            text = chunk.get("text", "").strip().replace("\n", " ")
            if not text:
                continue
            excerpts.append(f"- \"{text[:280]}\"" + ("...\"" if len(text) > 280 else "\""))
        return "\n".join(excerpts)
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            ref_type = 'Slide' if doc_type == 'pptx' else 'Page'
            context_parts.append(f"Source {i+1} ({ref_type} {chunk['metadata'].get('page', 'Unknown')}):\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_openai_answer(self, question: str, context: str, excerpts: str, reference: str) -> str:
        prompt = f"""You are an AI teaching assistant.

You have TWO jobs:
1) Grounding: use the provided lecture excerpts as the trusted reference.
2) Expansion: write an exam-ready explanation that is consistent with the lecture content.

IMPORTANT RULES:
- Do not invent page/slide numbers. Use ONLY the provided reference string.
- You MAY add general conceptual clarity and examples, but you must not contradict the lecture content.
- If the lecture context is insufficient, say so and keep the explanation limited.

Retrieved lecture excerpts (use these for quoting/highlighting):
{excerpts}

Full retrieved context (for details):
{context}

Student Question:
{question}

Write the response in EXACTLY this format:

Answer:
<Well-structured, exam-ready explanation. Use headings/numbered points if helpful.>

Reference:
{reference}

Optional:
Key Points:
- Point 1
- Point 2
- Point 3

Also include a small section inside the Answer called "Used from lecture" where you quote 1-3 short lines from the excerpts verbatim.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_simple_answer(self, question: str, context: str, excerpts: str, reference: str) -> str:
        # Fallback when no LLM is available: still structured and strictly based on retrieved content.
        if not context.strip():
            return (
                "Answer:\n"
                "I couldn't find relevant information in your lecture notes to answer this question.\n\n"
                f"Reference:\n{reference}"
            )

        return (
            "Answer:\n"
            "Below is an exam-ready explanation based on the retrieved lecture notes.\n\n"
            "Explanation (from lecture):\n"
            f"{context[:900]}" + ("..." if len(context) > 900 else "") + "\n\n"
            "Used from lecture:\n"
            f"{excerpts}\n\n"
            f"Reference:\n{reference}"
        )
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for chunk in chunks:
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            ref_type = 'Slide' if doc_type == 'pptx' else 'Page'
            source = {
                "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                "page": chunk['metadata'].get('page', 'Unknown'),
                "document": chunk['metadata'].get('source', 'Unknown'),
                "doc_type": doc_type,
                "reference_type": ref_type,
                "similarity_score": chunk.get('similarity_score', 0.0)
            }
            sources.append(source)
        return sources
    
    def clear_documents(self):
        self.vector_store.clear()
    
    def get_document_count(self) -> int:
        if self.vector_store.vector_db_type == "chroma":
            return self.vector_store.collection.count()
        else:
            return len(self.vector_store.chunks)
