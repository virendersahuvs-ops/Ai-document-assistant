from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_index(chunks: list[dict]):
    """Build a TF-IDF index over chunk text."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    return {"vectorizer": vectorizer, "matrix": matrix}

def retrieve(question: str, chunks: list[dict], index: dict, top_k: int = 4) -> list[dict]:
    """Return top matching chunks with cosine similarity scores."""
    if not question.strip():
        return []
    query_vector = index["vectorizer"].transform([question])
    scores = cosine_similarity(query_vector, index["matrix"]).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in ranked
        if scores[i] > 0
    ]
