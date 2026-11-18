"""
ingestion/ingest.py
───────────────────
Loads PDFs, chunks them with rich metadata (page, source, section),
then builds the hybrid FAISS+BM25 index.

Drop your PDFs in the  data/  folder and call:
    python -m ingestion.ingest
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple

from loguru import logger

from ingestion.mineru_parser import parse_pdf
from ingestion.chunking import chunk_document

from vectorstore.store import build_index


DATA_DIR       = Path("data")
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 140
MIN_CHUNK_SIZE = 250
MAX_CHUNK_SIZE = 1000


def load_pdf_chunks(pdf_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse with MinerU; split body hierarchically with intra-paragraph sentence overlap and keep each table atomic."""
    pdf_path = Path(pdf_path)
    return chunk_document(
        parse_pdf(pdf_path),
        pdf_path.name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE,
    )


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
