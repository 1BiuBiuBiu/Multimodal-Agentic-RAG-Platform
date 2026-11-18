"""Chunk browsing and MinerU previews; no embeddings or answer generation."""

import base64
import mimetypes
import pickle
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from ingestion.ingest import CHUNK_OVERLAP, CHUNK_SIZE, load_pdf_chunks
from ingestion.mineru_parser import MinerUError, MinerUUnavailable, read_cached_pdf
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


def _asset_images(element, artifact_dir):
    """Self-contained previews also survive temporary upload cleanup."""
    if not artifact_dir:
        return []
    root = Path(artifact_dir).resolve()
    paths = element.get('img_paths') or ([element['img_path']] if element.get('img_path') else [])
    images = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {'image/png','image/jpeg','image/webp','image/gif'}:
            continue
        images.append(f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii'))
    return images


def _preview_cache(path):
    try:
        return read_cached_pdf(path)
    except (OSError, ValueError, TypeError):
        logger.warning('Cannot read optional extraction preview cache: {}', path.name)
        return None


def _lines(value):
    return [value] if isinstance(value,str) else value or []


def _chunk_response(filename, texts, metas, *, indexed, parsed=None):
    chunks = [
        {**{key: value for key, value in meta.items() if key != "artifact_dir"},
         "id": i, "chunk_id": f"{filename}::{i}", "page": meta["page"],
         "text": text, "char_count": len(text)}
        for i, (text, meta) in enumerate(zip(texts, metas), start=1)
    ]
    for chunk, meta in zip(chunks, metas):
        if meta.get('type') in ('table', 'image'):
            chunk['asset_images'] = _asset_images(meta, meta.get('artifact_dir'))
    images = []
    indexed_img_paths = {meta.get('img_path') for meta in metas if meta.get('type') == 'image'}
    for i, element in enumerate((parsed or {}).get('images', []), start=1):
        text = '\n\n'.join(_lines(element.get('caption')) + _lines(element.get('footnote')))
        images.append({**element, 'id':i, 'source':filename, 'page_end':element['page'],
                       'text':text, 'char_count':len(text),
                       'indexed': element.get('img_path', '') in indexed_img_paths,
                       'asset_images':_asset_images(element, parsed.get('artifact_dir'))})
    return {
        "filename": filename, "indexed": indexed, "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "chunks": chunks,
        'images':images, 'total_images':len(images), 'extraction_available':parsed is not None,
    }


def _parse_pdf(path):
    try:
        return load_pdf_chunks(path)
    except MinerUUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MinerUError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        return _chunk_response(filename, selected_texts, selected_metas, indexed=True, parsed=_preview_cache(path))
    texts, metas = _parse_pdf(path)
    return _chunk_response(filename, texts, metas, indexed=False, parsed=_preview_cache(path))


@router.post("/preview")
def preview_pdf(file: UploadFile = File(...)):
    filename = _pdf_name(file.filename or "")
    with TemporaryDirectory(prefix="rag-pdf-preview-") as folder:
        path = Path(folder) / filename
        path.write_bytes(file.file.read())
        texts, metas = _parse_pdf(path)
        return _chunk_response(filename, texts, metas, indexed=False, parsed=_preview_cache(path))
