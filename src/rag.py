import json
import numpy as np
import faiss
import streamlit as st
from aws_clients import get_bedrock_runtime


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    client = get_bedrock_runtime()
    resp = client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]


def build_index(chunks: list[str]) -> faiss.IndexFlatL2:
    progress = st.progress(0, text="Embedding chunks…")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(embed_text(chunk))
        progress.progress((i + 1) / len(chunks), text=f"Embedding chunk {i+1}/{len(chunks)}")
    progress.empty()

    matrix = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index


def retrieve(question: str, index: faiss.IndexFlatL2, chunks: list[str], k: int = 3) -> list[str]:
    q_vec = np.array([embed_text(question)], dtype="float32")
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
