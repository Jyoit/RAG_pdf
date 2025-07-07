# RAG PDF Chatbot

This project is a **PDF-based Question Answering Chatbot** that uses Retrieval-Augmented Generation (RAG). It extracts text from a PDF, splits it into chunks, creates vector embeddings, stores them in a FAISS vector store, and answers user queries by retrieving relevant chunks and generating a response using Google Generative AI (Gemini).

## Features

- Upload and parse a PDF file.
- Split the PDF text into overlapping chunks.
- Create embeddings of chunks using Gemini embeddings.
- Store embeddings in a FAISS vector database.
- Perform similarity search over chunks for a given query.
- Generate answers using Gemini chat model based on retrieved context.

## Technologies Used

- Python
- LangChain
- Google Generative AI (`google-generativeai` and `langchain_google_genai`)
- FAISS
- PyPDF
- Streamlit (for the UI)


## Setup Instructions

### 1. Clone the repository

```bash```
git clone https://github.com/Jyoit/RAG_pdf.git
cd RAG_pdf

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the application

streamlit run server.py
