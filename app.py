import os
import streamlit as st
from document_processor import extract_text, chunk_text
from retriever import build_index, retrieve
from llm import generate_answer

st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Document Question-Answering Assistant")
st.caption("Ask questions about your documents. Answers are grounded in retrieved passages when an LLM API is configured.")

with st.sidebar:
    st.header("How it works")
    st.write("1. Upload PDF or TXT files.")
    st.write("2. The app extracts and chunks the text.")
    st.write("3. TF-IDF retrieval finds relevant passages.")
    st.write("4. If OPENAI_API_KEY is configured, an LLM answers using those passages.")
    st.divider()
    st.write("**Privacy note:** Documents are processed locally. If you configure an LLM API, retrieved passages are sent to that provider to generate answers.")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=6, value=4)
    min_score = st.slider("Minimum retrieval similarity", min_value=0.0, max_value=1.0, value=0.05, step=0.05)

uploads = st.file_uploader("Upload documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

if st.button("Process documents", type="primary", disabled=not uploads):
    all_chunks = []
    names = []
    errors = []
    for uploaded in uploads:
        try:
            raw_text = extract_text(uploaded.name, uploaded.getvalue())
            pieces = chunk_text(raw_text)
            if not pieces:
                errors.append(f"{uploaded.name}: no extractable text found.")
                continue
            all_chunks.extend({"source": uploaded.name, "text": piece} for piece in pieces)
            names.append(uploaded.name)
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")
    if all_chunks:
        st.session_state.chunks = all_chunks
        st.session_state.index = build_index(all_chunks)
        st.session_state.doc_names = names
        st.success(f"Processed {len(names)} document(s) into {len(all_chunks)} text chunks.")
    for err in errors:
        st.warning(err)

if st.session_state.chunks and st.session_state.index is not None:
    st.info("Loaded documents: " + ", ".join(st.session_state.doc_names))
    question = st.text_input("Ask a question about your documents", placeholder="What are the main findings?")
    if st.button("Get answer", disabled=not question.strip()):
        results = retrieve(question, st.session_state.chunks, st.session_state.index, top_k=top_k)
        results = [r for r in results if r["score"] >= min_score]
        if not results:
            st.warning("I couldn't find relevant evidence in the uploaded documents. Try a different question or lower the similarity threshold.")
        else:
            answer, mode = generate_answer(question, results)
            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Answer mode: {mode}")
            st.subheader("Retrieved source passages")
            for i, item in enumerate(results, 1):
                with st.expander(f"{i}. {item['source']} — relevance {item['score']:.3f}"):
                    st.write(item["text"])
    with st.expander("Show processed chunk inventory"):
        st.write(f"Total chunks: {len(st.session_state.chunks)}")
        for name in st.session_state.doc_names:
            count = sum(1 for chunk in st.session_state.chunks if chunk["source"] == name)
            st.write(f"- {name}: {count} chunks")
else:
    st.write("Upload one or more documents, then click **Process documents** to begin.")
