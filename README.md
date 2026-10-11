# AI Document Question-Answering Assistant

A Python-based application that enables users to ask questions about PDF and TXT documents using document retrieval and a locally running Large Language Model (LLM).

## Features

- PDF and TXT document processing
- Text extraction and overlapping chunking
- TF-IDF-based document retrieval
- Local answer generation using Ollama and Qwen 2.5 3B
- Streamlit user interface
- Retrieved source passages for reference
- Basic automated tests for document processing and retrieval
- Fallback handling when the local LLM is unavailable

## Tech Stack

Python, Streamlit, scikit-learn, TF-IDF, PyPDF, Ollama, Qwen 2.5 3B, unittest

## How It Works

1. Upload a PDF or TXT document.
2. Extract and divide the document text into chunks.
3. Retrieve relevant passages using TF-IDF and cosine similarity.
4. Pass the retrieved context and user question to the local LLM.
5. Display the generated answer and retrieved evidence.

## Future Improvements

- Dense embedding-based retrieval
- Retrieval and answer-quality evaluation
- Improved handling of unsupported questions
- Expanded automated tests
- Deployment and performance optimisation