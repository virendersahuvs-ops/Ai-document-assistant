# AI Document Question-Answering Assistant

A Python-based application that lets users ask questions about PDF and TXT documents using document retrieval and a locally running Large Language Model (LLM).

## Features

- PDF and TXT document processing
- Text extraction and overlapping chunking
- TF-IDF-based retrieval using cosine similarity
- Local answer generation using Ollama and Qwen 2.5 3B
- Interactive Streamlit interface
- Display of retrieved source passages
- Fallback handling when the local LLM is unavailable
- Automated unit tests for document processing and retrieval

## Tech Stack

Python, Streamlit, scikit-learn, TF-IDF, PyPDF, Ollama, Qwen 2.5 3B, unittest

## How It Works

1. Upload a PDF or TXT document.
2. Extract and split its text into chunks.
3. Retrieve relevant passages using TF-IDF similarity.
4. Send the retrieved context and question to the local LLM.
5. Display the generated answer and retrieved evidence.

## Setup

Install Python, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Ollama from https://ollama.com/download and download the model:

```bash
ollama run qwen2.5:3b
```

Start the application:

```bash
python -m streamlit run app.py
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Limitations

- Retrieval uses TF-IDF rather than dense semantic embeddings.
- Scanned PDFs requiring OCR are not supported.
- Answer quality depends on retrieved context and LLM behaviour.
- This is a portfolio prototype, not a production-secure application.

## Future Improvements

- Dense embedding-based retrieval
- Retrieval and answer-quality evaluation
- Improved handling of unsupported questions
- Expanded automated tests and deployment