"""Durable, bounded-parallel ablations against one frozen retrieval snapshot."""
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pipeline_config import PipelineConfig
from app.retrieval import TranslationCache, retrieve_documents
from app.llm import create_llm
from vectorstore.store import load_snapshot

RUN_DIR = Path("evaluation/runs")
VARIANTS_FILE = Path("evaluation/variants.json")
LOCK = threading.RLock()
JOBS = {}


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=10000)
    ground_truth: str = Field(min_length=1, max_length=30000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    config: PipelineConfig = Field(default_factory=PipelineConfig)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variants: list[Variant] = Field(min_length=1, max_length=10)
    dataset: list[EvalQuestion] = Field(min_length=1, max_length=100)
    scope: Literal["retrieval", "full"] = "retrieval"
    concurrency: int = Field(default=2, ge=1, le=5)
    use_ragas: bool = True

    @model_validator(mode="after")
    def comparable(self):
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("变体名称不能重复。")
        budgets = {(v.config.top_k, v.config.candidate_k) for v in self.variants}
        if len(budgets) != 1:
            raise ValueError("同批对比的 top_k 和 candidate_k 必须相同。")
        if not self.use_ragas and not any(q.relevant_chunk_ids for q in self.dataset):
            raise ValueError("关闭 RAGAs 时，至少为一个问题标注相关块 ID，才能计算检索指标。")
        return self


def label_metrics(retrieved_ids, relevant_ids, k):
    """Exact chunk-label metrics; unlabelled questions have no scores."""
    truth = set(relevant_ids)
    if not truth:
        return {}
    ranked = list(dict.fromkeys(retrieved_ids))[:k]
    flags = [int(item in truth) for item in ranked]
    hits = sum(flags)
    first = next((1 / (i + 1) for i, flag in enumerate(flags) if flag), 0.0)
    dcg = sum(flag / math.log2(i + 2) for i, flag in enumerate(flags))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(truth))))
    return {"hit_at_k": float(hits > 0), "precision_at_k": hits / k,
            "recall_at_k": hits / len(truth), "mrr": first, "ndcg_at_k": dcg / ideal}


def _judge(samples, scope):
    # Every worker owns fresh metric instances: no global RAGAs metric mutation.
    from datasets import Dataset
    from ragas import evaluate
    from evaluation.judge import DeepSeekJudge
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig
    judge = DeepSeekJudge(create_llm(temperature=0))
    metrics = [LLMContextPrecisionWithReference(llm=judge), LLMContextRecall(llm=judge)]
    embeddings = None
    if scope == "full":
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_core.embeddings import Embeddings
        from embeddings.embedder import embed_texts, embed_query
        class LocalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return embed_texts(texts)
            def embed_query(self, text):
                return embed_query(text)
        embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())
        metrics += [Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=embeddings)]
    data = Dataset.from_list([{
        "question": row["question"], "ground_truth": row["ground_truth"],
        "contexts": [c["text"] for c in row["sources"]] or row.get("web_contexts", []), "answer": row.get("answer") or "",
    } for row in samples])
    result = evaluate(data, metrics=metrics, llm=judge, embeddings=embeddings,
                      run_config=RunConfig(max_workers=1, timeout=120, max_retries=2),
                      raise_exceptions=False, show_progress=False)
    frame = result.to_pandas()
    values = []
    for _, row in frame.iterrows():
        values.append({metric.name: float(row[metric.name]) if math.isfinite(float(row[metric.name])) else None
                       for metric in metrics})
    return values


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _save(job):
    _write(RUN_DIR / f"{job['id']}.json", job)


def _error(exc):
    text = f"{type(exc).__name__}: {str(exc)}"
    for key in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY"):
        secret = os.getenv(key)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?:sk-|ghp_|tvly-)[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:300]


def _progress(job_id, name, **changes):
    with LOCK:
        JOBS[job_id]["results"][name].update(copy.deepcopy(changes))
        _save(JOBS[job_id])


def _run_variant(job_id, variant, request, snapshot, cache, q_workers: int = 1):
    samples = []
    try:
        # Load snapshot matching the variant's multimodal configuration
        variant_snapshot = snapshot if getattr(snapshot, "multimodal", True) == variant.config.multimodal else load_snapshot(multimodal=variant.config.multimodal)
        query_llm = create_llm(temperature=0)

        def _evaluate_single_question(item):
            idx, question = item
            started = time.perf_counter()
            if request.scope == "retrieval":
                result = retrieve_documents(question.question, variant.config, llm=query_llm,
                                            snapshot=variant_snapshot, translation_cache=cache)
                trace = [{"retrieval_query": result["retrieval_query"], "queries": result["queries"],
                          "snapshot_id": result["snapshot_id"]}]
                answer = None
                used_web = False
                retrieval_count = 1
                total_tokens = 0
            else:
                from app.agent import ask
                result = ask(question.question, config=variant.config, snapshot=variant_snapshot, translation_cache=cache)
                trace, answer, used_web = result["retrieval_trace"], result["answer"], result["used_web"]
                retrieval_count = len(trace) if trace else (result.get("rewrite_count", 0) + 1)
                total_tokens = result.get("total_tokens", 0)
            ids = [c["chunk_id"] for c in result["sources"]]
            return idx, {
                **question.model_dump(),
                "sources": result["sources"],
                "retrieved_chunk_ids": ids,
                "trace": trace,
                "answer": answer,
                "used_web": used_web,
                "retrieval_count": retrieval_count,
                "total_tokens": total_tokens,
                "web_contexts": result.get("web_contexts", []),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "metrics": label_metrics(ids, question.relevant_chunk_ids, variant.config.top_k)
            }

        ordered_samples = [None] * len(request.dataset)
        completed_count = 0
        samples_lock = threading.Lock()

        if q_workers > 1:
            _progress(job_id, variant.name, status="running", phase=f"并发评估中(并发度={q_workers})", completed_questions=0)
            with ThreadPoolExecutor(max_workers=q_workers, thread_name_prefix=f"eval-q-{variant.name}") as q_exec:
                futures = [q_exec.submit(_evaluate_single_question, (i, q)) for i, q in enumerate(request.dataset)]
                for fut in as_completed(futures):
                    idx, sample_data = fut.result()
                    with samples_lock:
                        ordered_samples[idx] = sample_data
                        completed_count += 1
                        curr_samples = [s for s in ordered_samples if s is not None]
                        _progress(job_id, variant.name, status="running",
                                  phase=f"并发评估中({completed_count}/{len(request.dataset)})",
                                  completed_questions=completed_count, samples=curr_samples)
        else:
            for index, question in enumerate(request.dataset):
                _progress(job_id, variant.name, status="running", phase="检索中", completed_questions=index)
                _, sample_data = _evaluate_single_question((index, question))
                ordered_samples[index] = sample_data
                _progress(job_id, variant.name, completed_questions=index + 1,
                          samples=[s for s in ordered_samples if s is not None])

        samples = ordered_samples

        if request.use_ragas:
            _progress(job_id, variant.name, phase="RAGAs 评分中")
            judged = _judge(samples, request.scope)
            if len(judged) != len(samples):
                raise ValueError("RAGAs 返回的评分数量与问题数量不一致。")
            for row, scores in zip(samples, judged):
                row["metrics"].update(scores)
        metric_names = sorted({name for row in samples for name in row["metrics"]})
        summary, valid_counts = {}, {}
        for name in metric_names:
            values = [row["metrics"].get(name) for row in samples if name in row["metrics"]]
            valid = [value for value in values if value is not None and math.isfinite(value)]
            valid_counts[name] = len(valid)
            # Do not silently average a different subset when judge scoring fails.
            summary[name] = round(sum(valid) / len(valid), 5) if valid and len(valid) == len(values) else None
        summary["latency_ms"] = round(sum(row["latency_ms"] for row in samples) / len(samples), 2)
        summary["avg_retrieval_count"] = round(sum(row.get("retrieval_count", 1) for row in samples) / len(samples), 2)
        summary["avg_tokens"] = round(sum(row.get("total_tokens", 0) for row in samples) / len(samples), 1)
        _progress(job_id, variant.name, status="completed", phase="完成", samples=samples,
                  metrics=summary, valid_counts=valid_counts,
                  labelled_questions=sum(bool(row.relevant_chunk_ids) for row in request.dataset))
    except Exception as exc:
        from loguru import logger
        logger.exception(f"Evaluation variant failed: {variant.name}")
        _progress(job_id, variant.name, status="failed", phase="失败", samples=samples,
                  error=_error(exc), metrics={})


def _run_batch(job_id, request):
    try:
        snapshot = load_snapshot()
        known_ids = set(snapshot.chunk_ids)
        unknown = {item for row in request.dataset for item in row.relevant_chunk_ids} - known_ids
        if unknown:
            raise ValueError("标注包含当前索引不存在的块 ID：" + ", ".join(sorted(unknown)[:5]))
        from embeddings.embedder import get_embedder, get_reranker, MODEL_LOCK
        warm_started = time.perf_counter()
        with MODEL_LOCK:
            if any(v.config.dense for v in request.variants):
                get_embedder()
            if any(v.config.rerank for v in request.variants):
                get_reranker()
            if any(getattr(v.config, "multimodal", True) for v in request.variants):
                from embeddings.visual_embedder import get_visual_embedder
                get_visual_embedder()
        with LOCK:
            JOBS[job_id].update(status="running", snapshot_id=snapshot.fingerprint,
                                corpus_chunks=len(snapshot.chunks), warmup_ms=round((time.perf_counter()-warm_started)*1000, 2))
            _save(JOBS[job_id])
        cache = TranslationCache()
        num_variants = len(request.variants)
        if num_variants == 1:
            variant_workers = 1
            question_workers = min(max(2, request.concurrency), len(request.dataset), 5)
        else:
            variant_workers = min(request.concurrency, num_variants)
            question_workers = max(1, request.concurrency // num_variants)

        with ThreadPoolExecutor(max_workers=variant_workers, thread_name_prefix="ablation-var") as executor:
            futures = [executor.submit(_run_variant, job_id, variant, request, snapshot, cache, question_workers)
                       for variant in request.variants]
            for future in as_completed(futures):
                future.result()
        with LOCK:
            job = JOBS[job_id]
            job["status"] = "completed" if all(r["status"] == "completed" for r in job["results"].values()) else "completed_with_errors"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save(job)
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(status="failed", error=_error(exc))
            _save(JOBS[job_id])


def start_batch(request):
    with LOCK:
        if any(job["status"] in ("queued", "running") for job in JOBS.values()):
            raise RuntimeError("已有一批评估正在运行，请等待完成后再启动。")
        job_id = uuid.uuid4().hex
        dataset_json = json.dumps([q.model_dump() for q in request.dataset], sort_keys=True, ensure_ascii=False)
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        JOBS[job_id] = {
            "id": job_id, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
            "request": request.model_dump(), "dataset_hash": hashlib.sha256(dataset_json.encode()).hexdigest(),
            "code_revision": git.stdout.strip(), "code_dirty": bool(dirty.stdout.strip()),
            "judge_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "results": {v.name: {"status": "queued", "phase": "等待执行", "completed_questions": 0,
                                  "metrics": {}, "samples": []} for v in request.variants},
        }
        _save(JOBS[job_id])
        threading.Thread(target=_run_batch, args=(job_id, request), daemon=True, name="evaluation-batch").start()
        return copy.deepcopy(JOBS[job_id])


def get_job(job_id):
    if not job_id.isalnum() or len(job_id) != 32:
        raise FileNotFoundError(job_id)
    with LOCK:
        if job_id in JOBS:
            return copy.deepcopy(JOBS[job_id])
        path = RUN_DIR / f"{job_id}.json"
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] in ("queued", "running"):
            job.update(status="interrupted", error="服务重启导致任务中断，请重新运行。")
            _save(job)
        return job
