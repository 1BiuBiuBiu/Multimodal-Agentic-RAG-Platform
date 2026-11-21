"""
embeddings/visual_embedder.py
──────────────────────────────
Multimodal embedding engine strictly powered by BAAI Visualized-BGE:
- Text/Tables: encoded via BGE-M3 -> 1024-dim dense vector
- Images: strictly encoded via Visualized_BGE(image=image_path, text=caption) -> 1024-dim dense vector
- Query: encoded to 1024-dim dense vector aligned in the unified multimodal space
- Baseline: pure-text BGE-M3 for ablation comparisons

Fail-fast principle: Never silently fall back to text-only degradation.
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import List, Dict, Any, Optional

from loguru import logger

from embeddings.embedder import embed_texts
from embeddings.model_downloader import verify_text_bge_model, ensure_visual_bge_weight
from visual_bge import Visualized_BGE

MODEL_LOCK = RLock()
DATA_MINERU_DIR = Path("data/.mineru")


def resolve_image_path(meta: Dict[str, Any]) -> Optional[Path]:
    """Resolve physical image file from chunk metadata."""
    if not isinstance(meta, dict):
        return None
    # 1. Explicit absolute/resolved path
    for key in ("image_path", "img_full_path", "image_file"):
        if meta.get(key):
            p = Path(meta[key])
            if p.is_file():
                return p

    # 2. Artifact dir + relative img_path
    rel = meta.get("img_path")
    if not rel:
        return None

    artifact_dir = meta.get("artifact_dir")
    if artifact_dir:
        p = Path(artifact_dir) / rel
        if p.is_file():
            return p

    # 3. Search in data/.mineru/*/<img_path>
    if DATA_MINERU_DIR.exists():
        for folder in DATA_MINERU_DIR.iterdir():
            if folder.is_dir():
                candidate = folder / rel
                if candidate.is_file():
                    return candidate

    # 4. Search in data/ or root
    candidate = Path("data") / rel
    if candidate.is_file():
        return candidate

    return None


class VisualBGEEmbedder:
    """
    Visualized-BGE multimodal encoder wrapper.
    Integrates BGE-M3 text backbone with vision projection.
    Strictly fail-fast: Never downgrades silently without images or model weights.
    """

    def __init__(self):
        # 1. Guarantee text BGE-M3 exists
        text_bge_path = verify_text_bge_model()
        logger.info(f"Verified BGE-M3 text model at: {text_bge_path}")

        # 2. Guarantee multimodal Visualized_m3.pth exists (warns and downloads if absent)
        visual_weight_path = ensure_visual_bge_weight()
        logger.info(f"Verified Visualized-BGE weights at: {visual_weight_path}")

        # 3. Load official Visualized_BGE model
        logger.info("Instantiating native Visualized_BGE multimodal model...")
        self.visual_model = Visualized_BGE(
            model_name_bge=str(text_bge_path),
            model_weight=str(visual_weight_path)
        )
        logger.success("Native Visualized_BGE model initialized and ready.")

    def encode_image_with_caption(self, image_path: Path, caption: str) -> List[float]:
        """
        Encode physical image pixels and accompanying caption strictly through Visualized_BGE.
        Raises FileNotFoundError if image_path does not exist on disk.
        """
        if not image_path.is_file():
            raise FileNotFoundError(f"Image chunk points to non-existent image file: {image_path}")

        import torch
        with torch.no_grad():
            vec = self.visual_model.encode(image=str(image_path), text=caption)
            if hasattr(vec, "cpu"):
                vec = vec.detach().cpu().numpy()
            if hasattr(vec, "flatten"):
                vec = vec.flatten().tolist()
            return vec

    def encode_query(self, query: str) -> List[float]:
        """Encode query text into the unified multimodal space."""
        import torch
        with torch.no_grad():
            vec = self.visual_model.encode(text=query)
            if hasattr(vec, "cpu"):
                vec = vec.detach().cpu().numpy()
            if hasattr(vec, "flatten"):
                vec = vec.flatten().tolist()
            return vec


@lru_cache(maxsize=1)
def get_visual_embedder() -> VisualBGEEmbedder:
    with MODEL_LOCK:
        return VisualBGEEmbedder()


def embed_multimodal_chunks(
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    multimodal: bool = True
) -> List[List[float]]:
    """
    Embed chunks and metadatas into 1024-dim vectors.
    - multimodal=True: Images strictly use Visualized_BGE (pixels + caption), Text/Table uses BGE-M3.
    - multimodal=False: Pure-text BGE-M3 on chunks alone (for ablation baseline).
    """
    if not chunks:
        return []

    if not multimodal:
        return embed_texts(chunks)

    visual_embedder = get_visual_embedder()
    results: List[List[float]] = [None] * len(chunks)  # type: ignore

    text_indices = []
    text_contents = []

    for i, (text, meta) in enumerate(zip(chunks, metadatas)):
        is_image = meta.get("type") == "image"
        img_path = resolve_image_path(meta) if is_image else None

        if is_image:
            if img_path is not None and img_path.is_file():
                caption = text
                results[i] = visual_embedder.encode_image_with_caption(img_path, caption)
            else:
                logger.warning(
                    f"Multimodal chunk specifies type='image' but physical image file could not be resolved. "
                    f"Falling back to text/caption embedding: {meta.get('chunk_id')}"
                )
                text_indices.append(i)
                text_contents.append(text)
        else:
            text_indices.append(i)
            text_contents.append(text)

    if text_contents:
        text_vectors = embed_texts(text_contents)
        for idx, vec in zip(text_indices, text_vectors):
            results[idx] = vec

    return results


def embed_multimodal_query(query: str, multimodal: bool = True) -> List[float]:
    """
    Embed user retrieval query into 1024-dim dense vector.
    - multimodal=True: encoded in the shared multimodal space via visual_embedder.encode_query.
    - multimodal=False: encoded via pure-text BGE-M3 embed_query.
    """
    if multimodal:
        return get_visual_embedder().encode_query(query)
    from embeddings.embedder import embed_query
    return embed_query(query)
