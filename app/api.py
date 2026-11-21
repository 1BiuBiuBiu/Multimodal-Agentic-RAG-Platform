"""
app/api.py
──────────
FastAPI production API — wraps the LangGraph agent.

Endpoints
─────────
  POST /ingest        upload a PDF and index it
  POST /query         ask a question, get an answer + sources
  POST /query/stream  stream status, answer tokens and final sources via SSE
  GET  /health        liveness probe (required for cloud deploy)
  GET  /metrics       RAGAs evaluation scores (if run)

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations
import json
import asyncio
from contextlib import suppress
import time
from pathlib import Path
from typing import List, Optional, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from app.pipeline_config import PipelineConfig

from app.agent import ask, astream_answer
from ingestion.ingest import load_pdf_chunks
from vectorstore.store import index_single_document
from app.documents import router as documents_router, _pdf_name
from app.experiment_api import router as experiment_router


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Document Assistant",
    description=(
        "Production RAG API with hybrid retrieval, "
        "cross-encoder reranking, LangGraph agentic loop, "
        "and Corrective RAG (query rewriting + web fallback)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(experiment_router)

DATA_DIR    = Path("data")
METRICS_FILE = Path("evaluation/latest_scores.json")
DATA_DIR.mkdir(exist_ok=True)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    history:    Optional[List[dict]] = None   # [{"role": "user"|"assistant", "content": "..."}]
    config: PipelineConfig = Field(default_factory=PipelineConfig)


class SourceItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    text:   str
    source: str
    page:   int
    chunk_id: Optional[str] = None
    type: Optional[str] = "text"
    asset_images: Optional[List[str]] = None
    caption: Optional[Any] = None
    section: Optional[str] = None
    page_end: Optional[int] = None
    mention_context: Optional[str] = None
    image_path: Optional[str] = None
    mentioned_visuals: Optional[List[dict]] = None


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[SourceItem]
    rewrite_count: int
    used_web:      bool
    latency_ms:    int
    retrieval_trace: list = Field(default_factory=list)
    pipeline_config: dict = Field(default_factory=dict)
    web_contexts: list[str] = Field(default_factory=list)
    degradation_warnings: List[str] = Field(default_factory=list)
    decision_trace: list = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    token_breakdown: list = Field(default_factory=list)
    total_tokens: int = 0
    query_doc_history: list = Field(default_factory=list)
    original_query: Optional[str] = ""


# ── Routes ───────────────────────────────────────────────────────────────────

def _history(req: QueryRequest):
    return [
        HumanMessage(content=m["content"]) if m.get("role") == "user"
        else AIMessage(content=m["content"])
        for m in (req.history or [])
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]

@app.get("/health")
def health():
    """Liveness probe — required for Docker/cloud deploy."""
    return {"status": "ok", "version": app.version}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and incrementally add/update it in the vector index.
    """
    filename = _pdf_name(Path(file.filename or "").name)

    dest = DATA_DIR / filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"Saved uploaded file: {dest}")

    try:
        chunks, metas = load_pdf_chunks(dest)
        stats = index_single_document(filename, chunks, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status":       "indexed",
        "filename":     filename,
        "chunks":       stats["doc_chunks"],
        "total_chunks": stats["total_chunks"],
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question. The LangGraph agent:
    1. Routes (documents vs LLM-only)
    2. Retrieves with hybrid FAISS+BM25
    3. Reranks with cross-encoder
    4. Grades relevance — rewrites query if needed (Corrective RAG)
    5. Falls back to web search if docs fail after max rewrites
    6. Generates a cited answer
    """
    t0 = time.time()
    try:
        result = ask(req.question, history=_history(req), config=req.config)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No index found. Upload a PDF via POST /ingest first."
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = int((time.time() - t0) * 1000)
    logger.info(f"Query answered in {latency}ms | rewrites={result['rewrite_count']}")

    return QueryResponse(
        answer        = result["answer"],
        sources       = [SourceItem(**s) for s in result["sources"]],
        rewrite_count = result["rewrite_count"],
        used_web      = result["used_web"],
        latency_ms    = latency,
        retrieval_trace = result.get("retrieval_trace", []),
        pipeline_config = result.get("pipeline_config", {}),
        web_contexts = result.get("web_contexts", []),
        degradation_warnings = result.get("degradation_warnings", []),
        decision_trace = result.get("decision_trace", []),
        token_usage = result.get("token_usage", {}),
        token_breakdown = result.get("token_breakdown", []),
        total_tokens = result.get("total_tokens", 0),
        query_doc_history = result.get("query_doc_history", []),
        original_query = result.get("original_query", req.question),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE: status/token events, then done with sources and performance metadata."""
    async def events():
        started = time.perf_counter()
        source = astream_answer(req.question, history=_history(req), config=req.config)
        pending = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(anext(source))
                completed, _ = await asyncio.wait({pending}, timeout=15)
                if not completed:
                    yield ": keep-alive\n\n"
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                pending = None
                if event["type"] == "done":
                    event["latency_ms"] = int((time.perf_counter() - started) * 1000)
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        except FileNotFoundError:
            yield "data: " + json.dumps({"type": "error", "content": "尚未建立文档索引，请先上传 PDF 并完成索引。"}, ensure_ascii=False) + "\n\n"
        except Exception:
            logger.exception("Streaming query failed")
            yield "data: " + json.dumps({"type": "error", "content": "回答生成失败，请检查后端日志、API 配置或网络后重试。"}, ensure_ascii=False) + "\n\n"
        finally:
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending
            await source.aclose()

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/metrics")
def metrics():
    """Return latest RAGAs evaluation scores if available."""
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    return {"detail": "No evaluation scores yet. Run: python -m evaluation.evaluate"}
