"""
vectorstore/store.py
────────────────────
Hybrid retrieval: dense FAISS (Visual-BGE Multimodal / BGE-M3 Text) + sparse BM25
Results fused with Reciprocal Rank Fusion (RRF).

Supports:
- Multimodal index (Visual-BGE): images encoded with pixels + caption
- Text-only baseline index (BGE-M3): for ablation comparison experiments
- Sparse BM25: indexing text and image captions for high-precision keyword recall
"""

from __future__ import annotations
import json
import os
import pickle
import hashlib
from dataclasses import dataclass
from collections import Counter
from threading import RLock
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from embeddings.embedder import (
    embed_texts,
    embed_query,
    rerank,
    rerank_indices,
)
from embeddings.visual_embedder import (
    embed_multimodal_chunks,
    embed_multimodal_query,
    resolve_image_path,
)



STORE_DIR        = Path("vectorstore/index")
FAISS_PATH       = STORE_DIR / "faiss.index"        # Visual-BGE multimodal index
FAISS_TEXT_PATH  = STORE_DIR / "faiss_text.index"   # Pure text baseline index
META_PATH        = STORE_DIR / "metadata.pkl"
BM25_PATH        = STORE_DIR / "bm25.pkl"

EMBED_DIM   = 1024   # Output dimension (1024 for both BGE-M3 and Visual-BGE-M3)
TOP_K_FETCH = 20     # fetch more, rerank down to fewer
TOP_K_FINAL = 5      # chunks returned to the LLM
INDEX_LOCK = RLock()

RRF_K       = 60     # standard RRF constant


# ── Build / persist ──────────────────────────────────────────────────────────

def _enrich_metadatas(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure image chunks have resolved image_path attributes."""
    enriched = []
    for meta in metadatas:
        m = dict(meta)
        if m.get("type") == "image":
            resolved = resolve_image_path(m)
            if resolved:
                m["image_path"] = str(resolved.resolve())
        enriched.append(m)
    return enriched


def build_index(chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
    """
    Build both Visual-BGE multimodal index and BGE-M3 pure-text index + BM25 sparse index.
    """
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    metadatas = _enrich_metadatas(metadatas)

    logger.info(f"Building multimodal and text-baseline index for {len(chunks)} chunks …")

    # 1. Multimodal vectors (Visual-BGE: image pixels + caption, text/table: BGE-M3)
    mm_vectors = np.array(embed_multimodal_chunks(chunks, metadatas, multimodal=True), dtype="float32")

    # 2. Pure text vectors (BGE-M3 baseline on text/caption alone)
    text_vectors = np.array(embed_multimodal_chunks(chunks, metadatas, multimodal=False), dtype="float32")

    with INDEX_LOCK:
        # Multimodal FAISS index
        mm_index = faiss.IndexFlatL2(EMBED_DIM)
        mm_index.add(mm_vectors)
        faiss.write_index(mm_index, str(FAISS_PATH))

        # Text-only baseline FAISS index
        text_index = faiss.IndexFlatL2(EMBED_DIM)
        text_index.add(text_vectors)
        faiss.write_index(text_index, str(FAISS_TEXT_PATH))

        # BM25 sparse index
        tokenised = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenised)

        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

    logger.success(f"Multimodal and text indexes saved → {STORE_DIR}")


# ── Load ─────────────────────────────────────────────────────────────────────

def _load_index(multimodal: bool = True):
    with INDEX_LOCK:
        target_path = FAISS_PATH if multimodal else FAISS_TEXT_PATH
        if not target_path.exists():
            # Fallback to whichever FAISS index exists
            fallback = FAISS_PATH if FAISS_PATH.exists() else FAISS_TEXT_PATH
            if not fallback.exists():
                raise FileNotFoundError("No FAISS index found. Run ingestion first.")
            target_path = fallback

        index = faiss.read_index(str(target_path))
        with META_PATH.open("rb") as file:
            store = pickle.load(file)
        with BM25_PATH.open("rb") as file:
            bm25 = pickle.load(file)
        return index, store["chunks"], store["metadatas"], bm25, target_path


@dataclass(frozen=True)
class RetrievalSnapshot:
    index: Any
    chunks: tuple
    metadatas: tuple
    bm25: Any
    chunk_ids: tuple
    fingerprint: str
    multimodal: bool = True


def load_snapshot(multimodal: bool = True) -> RetrievalSnapshot:
    with INDEX_LOCK:
        index, chunks, metas, bm25, index_path = _load_index(multimodal=multimodal)
        counts = Counter()
        ids = []
        for meta in metas:
            source = meta.get('source', 'doc')
            counts[source] += 1
            cid = meta.get('chunk_id') or f"{source}::{counts[source]}"
            ids.append(cid)
        digest = hashlib.sha256()
        for path in (index_path, META_PATH, BM25_PATH):
            if path.exists():
                digest.update(path.read_bytes())
        digest.update(b"mm" if multimodal else b"txt")
        return RetrievalSnapshot(
            index=index,
            chunks=tuple(chunks),
            metadatas=tuple(metas),
            bm25=bm25,
            chunk_ids=tuple(ids),
            fingerprint=digest.hexdigest(),
            multimodal=multimodal,
        )


def search_candidates(query, snapshot, config):
    """One query, optional dense (multimodal or text)/BM25 retrieval, no cross-encoder yet."""
    count = min(config.candidate_k, len(snapshot.chunks))
    if not count:
        return []
    rankings = []
    
    # Determine multimodal setting
    multimodal = getattr(config, "multimodal", True)
    
    # If snapshot doesn't match requested mode and alternative index file exists, switch index
    index_to_use = snapshot.index
    if getattr(snapshot, "multimodal", True) != multimodal:
        alt_path = FAISS_PATH if multimodal else FAISS_TEXT_PATH
        if alt_path.exists():
            with INDEX_LOCK:
                index_to_use = faiss.read_index(str(alt_path))

    if config.dense:
        vector = np.array([embed_multimodal_query(query, multimodal=multimodal)], dtype='float32')
        _, indexes = index_to_use.search(vector, count)
        rankings.append([int(i) for i in indexes[0] if 0 <= i < len(snapshot.chunks)])

    if config.bm25:
        scores = snapshot.bm25.get_scores(query.lower().split())
        rankings.append(np.argsort(-scores, kind='stable')[:count].tolist())

    fused = Counter()
    for ranking in rankings:
        for rank, index_id in enumerate(ranking):
            fused[index_id] += 1 / (RRF_K + rank + 1)
    return sorted(fused, key=lambda i: (-fused[i], i))[:count]


def hybrid_search(query: str, top_k: int = TOP_K_FINAL, multimodal: bool = True) -> List[Dict[str, Any]]:
    """Compatibility wrapper for single-query hybrid retrieval."""
    from app.pipeline_config import PipelineConfig
    snapshot = load_snapshot(multimodal=multimodal)
    config = PipelineConfig(top_k=top_k, candidate_k=max(TOP_K_FETCH, top_k), multi_query=False, multimodal=multimodal)
    ids = search_candidates(query, snapshot, config)
    order = rerank_indices(query, [snapshot.chunks[i] for i in ids], top_n=top_k) if ids else []
    return [{"text": snapshot.chunks[ids[i]], **snapshot.metadatas[ids[i]],
             "chunk_id": snapshot.chunk_ids[ids[i]]} for i in order]
