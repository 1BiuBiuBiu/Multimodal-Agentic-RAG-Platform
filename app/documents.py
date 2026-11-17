"""Read-only chunk browsing and PDF preview endpoints; no model inference."""

import pickle
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from ingestion.ingest import CHUNK_OVERLAP, CHUNK_SIZE, load_pdf_chunks
from vectorstore.store import META_PATH

router = APIRouter(prefix="/documents", tags=["Documents"])
DATA_DIR = Path("data")


def _pdf_name(filename: str) -> str:
    if not filename or any(char in filename for char in "/\\:\0") or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="请选择有效的 PDF 文件名。")
    return filename


def _stored_chunks():
    if not META_PATH.exists():
        return [], [], 0
    try:
        with META_PATH.open("rb") as file:
            store = pickle.load(file)
        if len(store["chunks"]) != len(store["metadatas"]):
            raise ValueError("Chunk metadata count mismatch")
        return store["chunks"], store["metadatas"], META_PATH.stat().st_mtime_ns
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError, ValueError):
        logger.exception("Cannot read indexed chunk metadata")
        raise HTTPException(status_code=503, detail="文档索引暂时不可读，请等待索引完成，或重新建立索引。")


def _chunk_response(filename, texts, metas, *, indexed):
    chunks = [
        {"id": i, "page": meta["page"], "text": text, "char_count": len(text)}
        for i, (text, meta) in enumerate(zip(texts, metas), start=1)
    ]
    return {
        "filename": filename, "indexed": indexed, "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "chunks": chunks,
    }


def _parse_pdf(path):
    try:
        return load_pdf_chunks(path)
    except Exception:
        logger.exception("PDF chunk preview failed")
        raise HTTPException(status_code=422, detail="无法读取该 PDF，请检查文件是否损坏或加密。")


@router.get("")
def list_documents():
    _, metas, revision = _stored_chunks()
    counts = Counter(meta["source"] for meta in metas)
    files = sorted(DATA_DIR.iterdir(), key=lambda path: path.name.casefold()) if DATA_DIR.exists() else []
    return {"documents": [
        {"filename": path.name, "size_bytes": path.stat().st_size,
         "indexed_chunks": counts[path.name], "revision": f"{path.stat().st_mtime_ns}:{revision}"}
        for path in files
        if path.is_file() and path.suffix.lower() == ".pdf" and path.resolve().parent == DATA_DIR.resolve()
    ]}


@router.get("/chunks")
def document_chunks(filename: str):
    path = DATA_DIR / _pdf_name(filename)
    if path.resolve().parent != DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="无效的 PDF 路径。")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF 文件不存在，请刷新文档列表。")
    texts, metas, _ = _stored_chunks()
    selected = [(text, meta) for text, meta in zip(texts, metas) if meta["source"] == filename]
    if selected:
        selected_texts, selected_metas = zip(*selected)
        return _chunk_response(filename, selected_texts, selected_metas, indexed=True)
    texts, metas = _parse_pdf(path)
    return _chunk_response(filename, texts, metas, indexed=False)


@router.post("/preview")
def preview_pdf(file: UploadFile = File(...)):
    filename = _pdf_name(file.filename or "")
    with TemporaryDirectory(prefix="rag-pdf-preview-") as folder:
        path = Path(folder) / filename
        path.write_bytes(file.file.read())
        texts, metas = _parse_pdf(path)
    return _chunk_response(filename, texts, metas, indexed=False)
