"""
ingestion/ingest.py
───────────────────
Loads PDFs, chunks them with rich metadata (page, source, section),
then builds the hybrid FAISS+BM25 index.

Drop your PDFs in the  data/  folder and call:
    python -m ingestion.ingest
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from vectorstore.store import build_index


DATA_DIR     = Path("data")
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 150


def load_pdf_chunks(pdf_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Split one PDF using the same settings as ingestion and chunk previews."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, metadatas = [], []
    for doc in PyPDFLoader(str(pdf_path)).load():
        page_num = doc.metadata.get("page", 0) + 1
        for chunk in splitter.split_text(doc.page_content):
            if len(chunk.strip()) < 30:
                continue
            chunks.append(chunk)
            metadatas.append({"source": pdf_path.name, "page": page_num})
    return chunks, metadatas


def load_pdfs(data_dir: Path = DATA_DIR) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load every PDF with page/source metadata for the hybrid index."""
    all_chunks, all_metas = [], []
    pdf_files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf") if data_dir.exists() else []
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_dir}. Add files and re-run.")
    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        chunks, metas = load_pdf_chunks(pdf_path)
        all_chunks.extend(chunks)
        all_metas.extend(metas)
    logger.success(f"Loaded {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)")
    return all_chunks, all_metas


def run_ingestion() -> None:
    chunks, metas = load_pdfs()
    if chunks:
        build_index(chunks, metas)
        logger.success("Ingestion complete. Index is ready.")


if __name__ == "__main__":
    run_ingestion()
