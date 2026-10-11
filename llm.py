
import ollama


def generate_answer(question: str, retrieved: list[dict]) -> tuple[str, str]:
    """
    Generate an answer using a local Ollama model and retrieved passages.
    Falls back to displaying retrieved evidence if generation fails.
    """

    if not retrieved:
        return (
            "I couldn't find relevant information in the uploaded documents.",
            "No relevant evidence",
        )

    context = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = f"""
You are an AI document question-answering assistant.

Rules:
1. Answer using only the supplied document context.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain enough evidence, clearly say so.
4. Mention the source filenames supporting your answer.
5. Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using supplied documents. "
                        "Be factual, cautious, and transparent about missing evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )

        answer = response["message"]["content"].strip()

        return answer, "Local LLM (Ollama: qwen2.5:3b)"

    except Exception as exc:
        # Preserve useful evidence if the local model is unavailable.
        evidence = "\n\n".join(
            f"Source: {item['source']}\n{item['text']}"
            for item in retrieved
        )

        return (
            "The local AI model could not generate an answer. "
            "Please check that Ollama is running and the model is installed.\n\n"
            f"Technical error: {type(exc).__name__}\n\n"
            f"Retrieved evidence for manual review:\n{evidence}",
            "Retrieval fallback",
        )
