"""
embeddings/embedder.py
──────────────────────
Upgrade from all-MiniLM-L6-v2  →  BAAI/bge-m3
  - State-of-the-art multilingual embedding model (2024/25)
  - Supports dense, sparse, and multi-vector retrieval
  - Ships a cross-encoder reranker for 2-stage retrieval
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import List

from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL_LOCK = RLock()

EMBED_MODEL = os.getenv("EMBED_MODEL_PATH") or "BAAI/bge-m3"
RERANK_MODEL = os.getenv("RERANK_MODEL_PATH") or "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Singleton: loads once, reused across all requests."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Singleton cross-encoder reranker."""
    logger.info(f"Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings → list of float vectors."""
    if not texts:
        return []
    with MODEL_LOCK:
        model = get_embedder()
        # Whole tables can be thousands of tokens long. Bound padded batch
        # size, rather than padding 32 short body chunks to one huge table.
        lengths = model.tokenizer(texts, padding=False, truncation=False,
                                  return_length=True)['length']
        batch_limit = max(1, int(os.getenv('EMBED_BATCH_SIZE', '8')))
        token_limit = max(1, int(os.getenv('EMBED_BATCH_TOKEN_LIMIT', '4096')))
        vectors = [None] * len(texts)
        batch = []
        completed = 0

        def encode_batch():
            nonlocal completed
            result = model.encode([texts[i] for i in batch], batch_size=len(batch),
                                  normalize_embeddings=True, show_progress_bar=False)
            for index, vector in zip(batch, result):
                vectors[index] = vector.tolist()
            previous = completed
            completed += len(batch)
            if len(texts) >= 100 and (completed == len(texts) or completed//100 != previous//100):
                logger.info(f"Embedded {completed}/{len(texts)} chunks")

        for index in sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True):
            if batch and (len(batch) >= batch_limit or lengths[batch[0]] * (len(batch)+1) > token_limit):
                encode_batch()
                batch.clear()
            batch.append(index)
        if batch:
            encode_batch()
    return vectors


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def rerank_scores(query: str, chunks: List[str]) -> List[float]:
    """Compute cross-encoder relevance scores for query and chunks."""
    if not chunks:
        return []
    with MODEL_LOCK:
        scores = get_reranker().predict([(query, chunk) for chunk in chunks])
    return [float(s) for s in scores]


def rerank_indices(query: str, chunks: List[str], top_n: int = 5) -> List[int]:
    """Return positions, preserving different sources that contain identical text."""
    if not chunks:
        return []
    scores = rerank_scores(query, chunks)
    return sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:top_n]


def rerank(query: str, chunks: List[str], top_n: int = 5) -> List[str]:
    return [chunks[i] for i in rerank_indices(query, chunks, top_n)]


