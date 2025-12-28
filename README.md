# AI Teaching Assistant for Lecture Notes

## Problem Statement
Students face difficulty finding specific answers from lengthy lecture notes and slides. Manual searching is time-consuming and inefficient, especially during exams and revisions.

## Project Description
This project builds an AI-powered Teaching Assistant using a Retrieval Augmented Generation (RAG) approach. Students can upload lecture notes and ask questions. The system retrieves relevant sections from the notes and generates accurate answers with exact page or slide references.

## System Architecture
- Document Upload (PDF/PPT/DOC)
- Text Extraction with page/slide metadata
- Chunking and Embedding generation
- Vector Database for retrieval
- LLM-based answer generation with references

## Features
- Question answering from lecture notes
- Exact text and equation highlighting
- Page and slide number references
- Fast and accurate responses
- Student-friendly interface

## Tech Stack
- Python
- Retrieval Augmented Generation (RAG)
- Vector Database (FAISS / ChromaDB)
- LLM (OpenAI / HuggingFace)
- Streamlit / Flask

## Project Structure
- data/
- embeddings/
- backend/
- frontend/
- README.md

## Setup Instructions
1. Install Python dependencies
2. Upload lecture notes
3. Run the application
4. Ask questions through the interface

## Sample Usage
Question: What is backpropagation?  
Answer: Backpropagation is an algorithm used to train neural networks.  
Reference: Slide 12

## Future Enhancements
- Multi-document support
- Voice-based queries
- Offline model support
- Improved equation rendering
  
**Step 1: Clone the Repository**
git clone <your-github-repository-link>
cd ai_teaching_assistant

**Step 2: Create a Virtual Environment**
python -m venv .venv

**Step 3: Activate the Virtual Environment**

Windows

.venv\Scripts\activate


Mac / Linux

source .venv/bin/activate

**Step 4: Install Required Dependencies**
pip install -r requirements.txt

**Step 5: Run the Streamlit Application**
python -m streamlit run app.py

**Step 6: Open the Application**

Open the browser and go to:

http://localhost:8501
