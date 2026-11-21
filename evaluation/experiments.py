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
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pipeline_config import PipelineConfig
from app.retrieval import TranslationCache, retrieve_documents
from app.llm import create_llm
from vectorstore.store import load_snapshot

RUN_DIR = Path("evaluation/runs")
VARIANTS_FILE = Path("evaluation/variants.json")
LOCK = threading.RLock()
JOBS = {}
CANCEL_EVENTS: dict[str, threading.Event] = {}


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=10000)
    ground_truth: str = Field(min_length=1, max_length=30000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    question_type: Literal["simple", "complex"] = "simple"


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    config: PipelineConfig = Field(default_factory=PipelineConfig)


RAGAS_METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "response_relevancy",
    "answer_correctness",
    "context_recall",
    "llm_context_recall",
    "context_precision",
    "llm_context_precision_with_reference",
)

ATOMIC_CONTEXT_RECALL_INSTRUCTION = (
    "Given a context, and an answer, first decompose the answer by its semantic content into "
    "independent, self-contained atomic factual statements or key points (do not merely split by "
    "periods; break apart compound sentences containing multiple facts, formulas, numbers, or "
    "clauses into separate atomic points). Then classify for each atomic statement whether it "
    "can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary "
    "classification. Output json with reason."
)

COMPLETENESS_CORRECTNESS_INSTRUCTION = (
    "Given ground truth statements and answer statements, evaluate the completeness and factual "
    "accuracy of the answer against the ground truth by classifying statements into: "
    "TP (true positive): core points/statements from the ground truth that are covered or supported "
    "by the answer (including equivalent phrasing or detailed elaboration); "
    "FN (false negative): core points/statements found in the ground truth that are missing or omitted "
    "from the answer; "
    "FP (false positive): ONLY statements in the answer that factually contradict, conflict with, or "
    "fabricate errors against the ground truth (do NOT penalize or classify as FP valid background "
    "explanations, derivations, or supporting details in a comprehensive answer that merely go beyond "
    "the concise ground truth summary). "
    "Each classified statement can only belong to one category. Provide a reason for each classification."
)

PARTIAL_SUPPORT_CONTEXT_PRECISION_INSTRUCTION = (
    "Given question, answer (reference), and a single retrieved context chunk, verify whether "
    "this context chunk was useful in arriving at the given answer. When the question or "
    "reference answer covers multiple aspects, sub-questions, tables, figures, or papers, "
    "mark verdict as 1 if the context provides ANY relevant key fact, number, definition, "
    "formula, table/figure evidence, or reasoning that supports at least part of the "
    "reference answer (including cross-lingual equivalence between English context and "
    "Chinese question/answer). Do NOT require a single chunk to cover the entire reference "
    "answer. Give verdict as \"1\" if useful and \"0\" if not with json output."
)


class _RagasProgressTracker:
    """Thread-safe tqdm-compatible progress hook passed via RAGAs evaluate(..., _pbar=...)."""

    def __init__(self, total: int, callback):
        self.total = max(0, int(total))
        self.n = 0
        self._callback = callback
        self._lock = threading.Lock()

    def update(self, n: int = 1):
        with self._lock:
            self.n = min(self.total, self.n + int(n)) if self.total else (self.n + int(n))
            current = self.n
            total = self.total
        if self._callback:
            self._callback(current, total)

    def reset(self, total=None):
        with self._lock:
            if total is not None:
                self.total = max(0, int(total))
            self.n = 0

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variants: list[Variant] = Field(min_length=1, max_length=10)
    dataset: list[EvalQuestion] = Field(min_length=1, max_length=100)
    scope: Literal["retrieval", "full"] = "retrieval"
    concurrency: int = Field(default=2, ge=1, le=5)
    ragas_concurrency: Optional[int] = Field(default=None, ge=1, le=64)
    use_ragas: bool = True
    defer_ragas: bool = False

    @model_validator(mode="after")
    def comparable(self):
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("变体名称不能重复。")
        budgets = {(v.config.top_k, v.config.candidate_k) for v in self.variants}
        if len(budgets) != 1:
            raise ValueError("同批对比的 top_k 和 candidate_k 必须相同。")
        if (
            not self.use_ragas
            and not self.defer_ragas
            and self.scope == "retrieval"
            and not any(q.relevant_chunk_ids for q in self.dataset)
        ):
            raise ValueError("关闭 RAGAs 时，至少为一个问题标注相关块 ID，才能计算检索指标。")
        return self


class RagasEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: list[str] = Field(min_length=1, max_length=20)
    ragas_concurrency: int = Field(default=16, ge=1, le=64)
    selected_metrics: Optional[list[str]] = None


def _is_job_ragas_scored(job: dict) -> bool:
    for v_res in (job.get("results") or {}).values():
        if not isinstance(v_res, dict):
            continue
        if v_res.get("ragas_scored") is True:
            return True
        metrics = v_res.get("metrics") or {}
        if any(k in metrics for k in RAGAS_METRIC_KEYS):
            return True
    return False


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


def _is_llm_only_sample(row: dict) -> bool:
    """Return True when the sample bypassed document/web retrieval and was answered by the LLM alone."""
    if row.get("route") == "llm_only":
        return True
    contexts = [c["text"] for c in row.get("sources", [])] or row.get("web_contexts", [])
    return not contexts and not row.get("trace")


def _format_chunk_for_judge(c: dict) -> str:
    """Build judge-visible text for a retrieved chunk including attached mention_context and visuals."""
    if not isinstance(c, dict):
        return str(c or "").strip()
    body = str(c.get("text") or "").strip()
    mention = str(c.get("mention_context") or "").strip()
    if mention and mention not in body:
        body = f"{body}\n{mention}" if body else mention
    for v in (c.get("mentioned_visuals") or []):
        if not isinstance(v, dict):
            continue
        raw_cap = v.get("caption")
        v_cap = " ".join(str(x) for x in raw_cap) if isinstance(raw_cap, list) else str(raw_cap or "")
        v_cap = v_cap.strip()
        v_text = str(v.get("text") or "").strip()
        v_mention = str(v.get("mention_context") or "").strip()
        extra = [p for p in (v_cap, v_text, v_mention) if p and p not in body]
        if extra:
            addition = "\n".join(extra)
            body = f"{body}\n{addition}" if body else addition
    return body


def _sample_full_contexts(row: dict) -> list[str]:
    sources_txt = [_format_chunk_for_judge(c) for c in (row.get("sources") or [])]
    sources_txt = [t for t in sources_txt if t]
    if sources_txt:
        return sources_txt
    return [str(w) for w in (row.get("web_contexts") or []) if str(w)]


def _extract_raw_subquery_context_groups(row: dict) -> list[list[str]]:
    """
    Extract raw, un-deduplicated ordered Top-K context lists grouped by (round, sub_query)
    from `trace[*]["sub_groups"]` for fair Context Precision scoring.
    Returns an empty list when legacy runs lack raw `sub_groups` (never splits deduplicated `sources`).
    """
    groups: list[list[str]] = []
    for trace_item in (row.get("trace") or []):
        if not isinstance(trace_item, dict):
            continue
        for grp in (trace_item.get("sub_groups") or []):
            if not isinstance(grp, dict):
                continue
            if grp.get("docs"):
                ctxs = [_format_chunk_for_judge(d) for d in grp["docs"]]
                ctxs = [t for t in ctxs if t]
            elif grp.get("contexts"):
                ctxs = [str(x).strip() for x in grp["contexts"] if str(x).strip()]
            else:
                ctxs = []
            if ctxs:
                groups.append(ctxs)
    return groups


def _extract_subquery_context_groups(row: dict) -> list[list[str]]:
    """
    Extract ordered context lists for Context Precision scoring.
    Uses raw un-deduplicated `trace[*]["sub_groups"]` when available; for legacy runs without
    raw `sub_groups`, falls back to full contexts as a single group rather than splitting
    deduplicated `sources` (which would hollow out later sub-queries).
    """
    groups = _extract_raw_subquery_context_groups(row)
    if groups:
        return groups
    full = _sample_full_contexts(row)
    return [full] if full else []


def _restore_missing_mention_chunks(samples: list[dict], snapshot=None) -> None:
    """Restore body text chunks referenced by table/image `mention_chunk_id` when older runs dropped them."""
    needs_restore = any(
        isinstance(c, dict) and c.get("mention_chunk_id")
        and c.get("mention_chunk_id") not in {s.get("chunk_id") for s in (row.get("sources") or []) if isinstance(s, dict)}
        for row in samples
        for c in (row.get("sources") or [])
    )
    if not needs_restore:
        return
    if snapshot is None:
        try:
            snapshot = load_snapshot()
        except Exception:
            return
    if not snapshot or not getattr(snapshot, "chunk_ids", None):
        return
    snap_map = {
        cid: (idx, txt, meta)
        for idx, (cid, txt, meta) in enumerate(zip(snapshot.chunk_ids, snapshot.chunks, snapshot.metadatas))
    }
    for row in samples:
        sources = row.get("sources") or []
        if not sources:
            continue
        existing_ids = {s.get("chunk_id") for s in sources if isinstance(s, dict) and s.get("chunk_id")}
        updated_sources = []
        changed = False
        for c in sources:
            mcid = c.get("mention_chunk_id") if isinstance(c, dict) else None
            if mcid and mcid not in existing_ids and mcid in snap_map:
                idx_s, txt, meta = snap_map[mcid]
                restored_chunk = {
                    **meta,
                    "text": txt,
                    "chunk_id": mcid,
                    "index_id": idx_s,
                    "sub_query": c.get("sub_query", ""),
                }
                updated_sources.append(restored_chunk)
                existing_ids.add(mcid)
                changed = True
            updated_sources.append(c)
        if changed:
            row["sources"] = updated_sources
            row["retrieved_chunk_ids"] = [s["chunk_id"] for s in updated_sources if isinstance(s, dict) and s.get("chunk_id")]


def _judge(samples, scope, max_workers: int = 1, progress_cb=None, selected_metrics: Optional[list[str]] = None):
    # Every worker owns fresh metric instances: no global RAGAs metric mutation.
    if not samples:
        return []
    from datasets import Dataset
    from ragas import evaluate
    from evaluation.judge import DeepSeekJudge
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
        AnswerCorrectness,
    )
    from ragas.run_config import RunConfig

    judge = DeepSeekJudge(create_llm(temperature=0, timeout=600, max_retries=1))
    workers = max(1, int(max_workers))
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

    def _build_metric(metric_name: str):
        if metric_name == "llm_context_precision_with_reference":
            prec_metric = LLMContextPrecisionWithReference(llm=judge)
            prec_metric.context_precision_prompt = copy.deepcopy(prec_metric.context_precision_prompt)
            prec_metric.context_precision_prompt.instruction = PARTIAL_SUPPORT_CONTEXT_PRECISION_INSTRUCTION
            return prec_metric
        if metric_name == "context_recall":
            recall_metric = LLMContextRecall(llm=judge)
            recall_metric.context_recall_prompt = copy.deepcopy(recall_metric.context_recall_prompt)
            recall_metric.context_recall_prompt.instruction = ATOMIC_CONTEXT_RECALL_INSTRUCTION
            return recall_metric
        if metric_name == "faithfulness":
            return Faithfulness(llm=judge)
        if metric_name == "answer_relevancy":
            return ResponseRelevancy(llm=judge, embeddings=embeddings)
        if metric_name == "answer_correctness":
            corr_metric = AnswerCorrectness(llm=judge, embeddings=embeddings)
            corr_metric.correctness_prompt = copy.deepcopy(corr_metric.correctness_prompt)
            corr_metric.correctness_prompt.instruction = COMPLETENESS_CORRECTNESS_INSTRUCTION
            return corr_metric
        raise ValueError(f"Unsupported RAGAs metric: {metric_name}")

    values: list[dict] = [{} for _ in samples]
    retrieved_batches: dict[tuple[str, ...], list[int]] = {}
    llm_only_batches: dict[tuple[str, ...], list[int]] = {}
    multi_prec_indices: list[int] = []
    groups_by_idx: dict[int, list[list[str]]] = {}

    for i, row in enumerate(samples):
        is_llm_only = scope == "full" and _is_llm_only_sample(row)
        if scope != "full":
            desired = ("llm_context_precision_with_reference", "context_recall")
        elif is_llm_only:
            desired = ("answer_correctness", "answer_relevancy")
        else:
            desired = (
                "llm_context_precision_with_reference",
                "context_recall",
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
            )

        if selected_metrics:
            selected_set = set(selected_metrics)
            needed = [m_name for m_name in desired if m_name in selected_set]
        else:
            needed = list(desired)

        if "llm_context_precision_with_reference" in needed:
            raw_sub_groups = _extract_raw_subquery_context_groups(row)
            if raw_sub_groups:
                groups_by_idx[i] = raw_sub_groups
                needed.remove("llm_context_precision_with_reference")
                multi_prec_indices.append(i)

        if needed:
            key = tuple(needed)
            target_map = llm_only_batches if is_llm_only else retrieved_batches
            target_map.setdefault(key, []).append(i)

    run_cfg = RunConfig(max_workers=workers, timeout=600, max_retries=1)
    total_steps = (
        sum(len(m_keys) * len(idxs) for m_keys, idxs in retrieved_batches.items())
        + sum(len(m_keys) * len(idxs) for m_keys, idxs in llm_only_batches.items())
        + sum(len(groups_by_idx[pos]) for pos in multi_prec_indices)
    )
    tracker = _RagasProgressTracker(total_steps, progress_cb) if (progress_cb is not None and total_steps > 0) else None
    if progress_cb is not None:
        progress_cb(0, total_steps)
    expected_done = 0

    def _run_batch_map(batch_map: dict[tuple[str, ...], list[int]]):
        nonlocal expected_done
        for metric_keys, indices in batch_map.items():
            metric_objs = [_build_metric(m_key) for m_key in metric_keys]
            data = Dataset.from_list([{
                "question": samples[idx]["question"],
                "ground_truth": samples[idx]["ground_truth"],
                "contexts": _sample_full_contexts(samples[idx]),
                "answer": samples[idx].get("answer") or "",
            } for idx in indices])
            eval_kwargs = dict(
                metrics=metric_objs,
                llm=judge,
                embeddings=embeddings,
                run_config=run_cfg,
                raise_exceptions=True,
                show_progress=False,
            )
            if tracker is not None:
                eval_kwargs["_pbar"] = tracker
            result = evaluate(data, **eval_kwargs)
            expected_done += len(metric_objs) * len(indices)
            if tracker is not None and tracker.n < expected_done:
                tracker.update(expected_done - tracker.n)
            frame = result.to_pandas()
            for pos, (_, frame_row) in zip(indices, frame.iterrows()):
                for metric in metric_objs:
                    raw_val = frame_row[metric.name]
                    if raw_val is None:
                        raise RuntimeError(f"RAGAs 指标 {metric.name} 未能从大模型获得有效数值结果（返回值为空或 NaN）。")
                    fval = float(raw_val)
                    if not math.isfinite(fval):
                        raise RuntimeError(f"RAGAs 指标 {metric.name} 未能从大模型获得有效数值结果（返回值为空或 NaN）。")
                    values[pos][metric.name] = fval

    _run_batch_map(retrieved_batches)
    _run_batch_map(llm_only_batches)

    # Evaluate Context Precision independently per (round, sub_query) Top-K group, then average per question.
    if multi_prec_indices:
        prec_rows = []
        offsets: list[tuple[int, int, int]] = []
        for pos in multi_prec_indices:
            row = samples[pos]
            start = len(prec_rows)
            for ctx_group in groups_by_idx[pos]:
                prec_rows.append({
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "contexts": ctx_group,
                    "answer": row.get("answer") or "",
                })
            offsets.append((pos, start, len(prec_rows)))

        prec_metric = _build_metric("llm_context_precision_with_reference")
        prec_kwargs = dict(
            metrics=[prec_metric],
            llm=judge,
            embeddings=embeddings,
            run_config=run_cfg,
            raise_exceptions=True,
            show_progress=False,
        )
        if tracker is not None:
            prec_kwargs["_pbar"] = tracker
        prec_result = evaluate(Dataset.from_list(prec_rows), **prec_kwargs)
        expected_done += len(prec_rows)
        if tracker is not None and tracker.n < expected_done:
            tracker.update(expected_done - tracker.n)
        prec_frame = prec_result.to_pandas()
        col_scores = list(prec_frame[prec_metric.name])
        for pos, start, end in offsets:
            group_scores = col_scores[start:end]
            parsed: list[float] = []
            for raw_v in group_scores:
                if raw_v is None:
                    raise RuntimeError("RAGAs 指标 llm_context_precision_with_reference 未能从大模型获得有效数值结果（存在分组返回空或 NaN）。")
                fval = float(raw_v)
                if not math.isfinite(fval):
                    raise RuntimeError("RAGAs 指标 llm_context_precision_with_reference 未能从大模型获得有效数值结果（存在分组返回空或 NaN）。")
                parsed.append(fval)
            if not parsed:
                raise RuntimeError("RAGAs 指标 llm_context_precision_with_reference 未能从大模型获得有效数值结果（存在分组返回空或 NaN）。")
            values[pos][prec_metric.name] = sum(parsed) / len(parsed)

    return values


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _save(job):
    _write(RUN_DIR / f"{job['id']}.json", job)


def _error(exc):
    raw = f"{type(exc).__name__}: {str(exc)}"
    cause = getattr(exc, "__cause__", None)
    if cause and str(cause) and str(cause) not in raw:
        raw = f"{raw} (caused by {type(cause).__name__}: {str(cause)})"
    for key in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY"):
        secret = os.getenv(key)
        if secret:
            raw = raw.replace(secret, "[REDACTED]")
    raw = re.sub(r"(?:sk-|ghp_|tvly-)[A-Za-z0-9_-]+", "[REDACTED]", raw)
    low = raw.lower()
    status_code = getattr(exc, "status_code", None)
    if (
        status_code == 402
        or "insufficient balance" in low
        or "insufficient_balance" in low
        or re.search(r"\b(?:error code:\s*|http\s*)402\b", low)
    ):
        return f"💰 DeepSeek API 账户余额不足 (HTTP 402: Insufficient Balance)，请充值后重新评分。详情：{raw}"[:600]
    if (
        status_code == 429
        or "ratelimiterror" in low
        or "too many requests" in low
        or "concurrency limit" in low
        or re.search(r"\b(?:error code:\s*|http\s*)429\b", low)
    ):
        return f"🚦 DeepSeek API 触发并发超限或限流 (HTTP 429)，请调低「RAGAs API 并发请求数」或充值提升并发额度后重试。详情：{raw}"[:600]
    if (
        status_code == 401
        or "authenticationerror" in low
        or "invalid_api_key" in low
        or re.search(r"\b(?:error code:\s*|http\s*)401\b", low)
    ):
        return f"🔑 DeepSeek API Key 鉴权失败 (HTTP 401)，请检查 .env 中的 DEEPSEEK_API_KEY。详情：{raw}"[:600]
    if (
        isinstance(exc, TimeoutError)
        or "timeouterror" in low
        or "apitimeouterror" in low
        or "readtimeout" in low
        or "timed out" in low
    ):
        return f"⏱️ DeepSeek API 响应超时（已等待 600 秒未返回），请检查网络或 API 服务状态后重试。详情：{raw}"[:600]
    return raw[:600]


def _is_cancelled(job_id: str) -> bool:
    ev = CANCEL_EVENTS.get(job_id)
    if ev and ev.is_set():
        return True
    with LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.get("status") == "cancelled")


def _progress(job_id, name, **changes):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if job.get("status") == "cancelled" and changes.get("status") in ("running", "completed", "failed"):
            return
        v_res = job["results"][name]
        if v_res.get("status") == "cancelled" and changes.get("status") in ("running", "completed", "failed"):
            return
        v_res.update(copy.deepcopy(changes))
        _save(job)


def _summarize_samples(subset: list[dict]) -> tuple[dict, dict]:
    """Aggregate RAGAs, chunk-label, retrieval rounds, tokens, and latency metrics over a sample subset."""
    if not subset:
        return {}, {}
    metric_names = sorted({name for row in subset for name in row.get("metrics", {})})
    summary, valid_counts = {}, {}
    for name in metric_names:
        values = [row["metrics"].get(name) for row in subset if name in row.get("metrics", {})]
        valid = [value for value in values if value is not None and math.isfinite(value)]
        valid_counts[name] = len(valid)
        # Do not silently average a different subset when judge scoring fails.
        summary[name] = round(sum(valid) / len(valid), 5) if valid and len(valid) == len(values) else None
    summary["question_count"] = len(subset)
    summary["latency_ms"] = round(sum(row["latency_ms"] for row in subset) / len(subset), 2)
    summary["avg_retrieval_count"] = round(sum(row.get("retrieval_count", 1) for row in subset) / len(subset), 2)
    summary["avg_tokens"] = round(sum(row.get("total_tokens", 0) for row in subset) / len(subset), 1)
    return summary, valid_counts


def _run_variant(job_id, variant, request, snapshot, cache, q_workers: int = 1):
    samples = []
    try:
        if _is_cancelled(job_id):
            return
        # Load snapshot matching the variant's multimodal configuration
        variant_snapshot = snapshot if getattr(snapshot, "multimodal", True) == variant.config.multimodal else load_snapshot(multimodal=variant.config.multimodal)
        query_llm = create_llm(temperature=0)

        def _evaluate_single_question(item):
            idx, question = item
            if _is_cancelled(job_id):
                return idx, None
            started = time.perf_counter()
            if request.scope == "retrieval":
                result = retrieve_documents(question.question, variant.config, llm=query_llm,
                                            snapshot=variant_snapshot, translation_cache=cache)
                trace = [{
                    "retrieval_query": result["retrieval_query"],
                    "queries": result["queries"],
                    "snapshot_id": result["snapshot_id"],
                    "sub_groups": result.get("sub_groups", []),
                }]
                answer = None
                used_web = False
                route = "documents"
                retrieval_count = 1
                total_tokens = int((result.get("token_usage") or {}).get("total_tokens", 0) or 0)
            else:
                from app.agent import ask
                result = ask(question.question, config=variant.config, snapshot=variant_snapshot, translation_cache=cache)
                trace, answer, used_web = result["retrieval_trace"], result["answer"], result["used_web"]
                route = result.get("route") or (
                    "llm_only"
                    if not trace and not result.get("sources") and not result.get("web_contexts")
                    else "documents"
                )
                retrieval_count = 0 if route == "llm_only" else (len(trace) if trace else (result.get("rewrite_count", 0) + 1))
                total_tokens = int(result.get("total_tokens", 0) or 0)
            ids = [c["chunk_id"] for c in result["sources"]]
            return idx, {
                **question.model_dump(),
                "route": route,
                "sources": result["sources"],
                "retrieved_chunk_ids": ids,
                "trace": trace,
                "answer": answer,
                "used_web": used_web,
                "retrieval_count": retrieval_count,
                "total_tokens": total_tokens,
                "web_contexts": result.get("web_contexts", []),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "metrics": {} if route == "llm_only" else label_metrics(ids, question.relevant_chunk_ids, variant.config.top_k)
            }

        ordered_samples = [None] * len(request.dataset)
        completed_count = 0
        samples_lock = threading.Lock()

        if q_workers > 1:
            _progress(job_id, variant.name, status="running", phase=f"并发评估中(并发度={q_workers})", completed_questions=0)
            q_exec = ThreadPoolExecutor(max_workers=q_workers, thread_name_prefix=f"eval-q-{variant.name}")
            try:
                futures = [q_exec.submit(_evaluate_single_question, (i, q)) for i, q in enumerate(request.dataset)]
                for fut in as_completed(futures):
                    if _is_cancelled(job_id):
                        for pending_fut in futures:
                            pending_fut.cancel()
                        return
                    idx, sample_data = fut.result()
                    if sample_data is None or _is_cancelled(job_id):
                        return
                    with samples_lock:
                        ordered_samples[idx] = sample_data
                        completed_count += 1
                        curr_samples = [s for s in ordered_samples if s is not None]
                        _progress(job_id, variant.name, status="running",
                                  phase=f"并发评估中({completed_count}/{len(request.dataset)})",
                                  completed_questions=completed_count, samples=curr_samples)
            finally:
                q_exec.shutdown(wait=not _is_cancelled(job_id), cancel_futures=True)
        else:
            for index, question in enumerate(request.dataset):
                if _is_cancelled(job_id):
                    return
                _progress(job_id, variant.name, status="running", phase="检索中", completed_questions=index)
                _, sample_data = _evaluate_single_question((index, question))
                if sample_data is None or _is_cancelled(job_id):
                    return
                ordered_samples[index] = sample_data
                _progress(job_id, variant.name, completed_questions=index + 1,
                          samples=[s for s in ordered_samples if s is not None])

        if _is_cancelled(job_id):
            return
        samples = ordered_samples

        if request.use_ragas:
            if _is_cancelled(job_id):
                return
            judge_workers = max(1, int(request.ragas_concurrency or request.concurrency))
            _progress(job_id, variant.name, phase=f"RAGAs 评分中(并发度={judge_workers})")
            last_save_ts = [0.0]

            def _on_variant_ragas_progress(done: int, total: int):
                pct = int((done / total) * 100) if total else 100
                now = time.monotonic()
                with LOCK:
                    job = JOBS.get(job_id)
                    if not job or job.get("status") == "cancelled":
                        return
                    v_res = job["results"].get(variant.name)
                    if not isinstance(v_res, dict) or v_res.get("status") == "cancelled":
                        return
                    v_res["ragas_completed_steps"] = done
                    v_res["ragas_total_steps"] = total
                    v_res["phase"] = f"RAGAs 评分中 ({done}/{total} · {pct}%)"
                    if done == 0 or done >= total or (now - last_save_ts[0]) >= 0.3:
                        _save(job)
                        last_save_ts[0] = now

            judged = _judge(samples, request.scope, max_workers=judge_workers, progress_cb=_on_variant_ragas_progress)
            if _is_cancelled(job_id):
                return
            if len(judged) != len(samples):
                raise ValueError("RAGAs 返回的评分数量与问题数量不一致。")
            for row, scores in zip(samples, judged):
                row["metrics"].update(scores)
        if _is_cancelled(job_id):
            return
        summary, valid_counts = _summarize_samples(samples)
        simple_samples = [row for row in samples if row.get("question_type", "simple") == "simple"]
        complex_samples = [row for row in samples if row.get("question_type") == "complex"]
        simple_summary, simple_valid = _summarize_samples(simple_samples)
        complex_summary, complex_valid = _summarize_samples(complex_samples)
        final_phase = "已生成结果(待RAGAs评分)" if (not request.use_ragas and getattr(request, "defer_ragas", False)) else "完成"
        _progress(
            job_id,
            variant.name,
            status="completed",
            phase=final_phase,
            ragas_scored=bool(request.use_ragas),
            samples=samples,
            metrics=summary,
            valid_counts=valid_counts,
            metrics_by_type={"overall": summary, "simple": simple_summary, "complex": complex_summary},
            valid_counts_by_type={"overall": valid_counts, "simple": simple_valid, "complex": complex_valid},
            labelled_questions=sum(bool(row.relevant_chunk_ids) for row in request.dataset),
        )
    except Exception as exc:
        if _is_cancelled(job_id):
            return
        from loguru import logger
        logger.exception(f"Evaluation variant failed: {variant.name}")
        _progress(job_id, variant.name, status="failed", phase="失败", samples=samples,
                  error=_error(exc), metrics={})


def _run_batch(job_id, request):
    try:
        if _is_cancelled(job_id):
            return
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
        if _is_cancelled(job_id):
            return
        with LOCK:
            if JOBS[job_id].get("status") == "cancelled":
                return
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

        executor = ThreadPoolExecutor(max_workers=variant_workers, thread_name_prefix="ablation-var")
        try:
            futures = [executor.submit(_run_variant, job_id, variant, request, snapshot, cache, question_workers)
                       for variant in request.variants]
            for future in as_completed(futures):
                if _is_cancelled(job_id):
                    for pending_fut in futures:
                        pending_fut.cancel()
                    return
                future.result()
        finally:
            executor.shutdown(wait=not _is_cancelled(job_id), cancel_futures=True)

        with LOCK:
            job = JOBS[job_id]
            if job.get("status") == "cancelled":
                return
            job["status"] = "completed" if all(r["status"] == "completed" for r in job["results"].values()) else "completed_with_errors"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save(job)
    except Exception as exc:
        with LOCK:
            if JOBS[job_id].get("status") == "cancelled":
                return
            JOBS[job_id].update(status="failed", error=_error(exc))
            _save(JOBS[job_id])


def start_batch(request):
    with LOCK:
        if any(job["status"] in ("queued", "running") for job in JOBS.values()):
            raise RuntimeError("已有一批评估正在运行，请等待完成后再启动。")
        job_id = uuid.uuid4().hex
        CANCEL_EVENTS[job_id] = threading.Event()
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


def _run_ragas_for_jobs(
    job_ids: list[str],
    ragas_concurrency: int = 16,
    selected_metrics: Optional[list[str]] = None,
):
    """Pool already-generated samples across one or more jobs and run high-concurrency RAGAs scoring."""
    try:
        grouped: dict[str, list[tuple[str, str, list[dict]]]] = {}
        try:
            snap_for_restore = load_snapshot()
        except Exception:
            snap_for_restore = None
        with LOCK:
            for jid in job_ids:
                if _is_cancelled(jid):
                    continue
                job = JOBS.get(jid)
                if not job:
                    continue
                scope = (job.get("request") or {}).get("scope", "retrieval")
                for v_name, v_res in (job.get("results") or {}).items():
                    v_samples = v_res.get("samples") or []
                    if v_samples:
                        copied = copy.deepcopy(v_samples)
                        _restore_missing_mention_chunks(copied, snapshot=snap_for_restore)
                        grouped.setdefault(scope, []).append((jid, v_name, copied))

        scored_updates: dict[tuple[str, str], list[dict]] = {}
        for scope, items in grouped.items():
            if any(_is_cancelled(jid) for jid in job_ids):
                break
            pooled_samples: list[dict] = []
            spans: list[tuple[str, str, int, int, list[dict]]] = []
            for jid, v_name, v_samples in items:
                start = len(pooled_samples)
                pooled_samples.extend(v_samples)
                end = len(pooled_samples)
                spans.append((jid, v_name, start, end, v_samples))
            if not pooled_samples:
                continue
            mode_word = "指定指标重评" if selected_metrics else "批量评分"
            last_save_ts = [0.0]

            def _on_batch_ragas_progress(done: int, total: int, _spans=spans, _total_pooled=len(pooled_samples), _mode=mode_word):
                pct = int((done / total) * 100) if total else 100
                now = time.monotonic()
                with LOCK:
                    dirty_jobs: dict[str, dict] = {}
                    # Distribute steps cleanly per (job_id, variant) so UI sums exact totals without double-counting.
                    allocated_done = 0
                    allocated_total = 0
                    for idx_s, (s_jid, s_vname, s_start, s_end, _) in enumerate(_spans):
                        if _is_cancelled(s_jid):
                            continue
                        job = JOBS.get(s_jid)
                        if not job or job.get("status") == "cancelled":
                            continue
                        v_res = (job.get("results") or {}).get(s_vname)
                        if not isinstance(v_res, dict) or v_res.get("status") == "cancelled":
                            continue
                        span_len = max(1, s_end - s_start)
                        if idx_s == len(_spans) - 1:
                            v_done = max(0, done - allocated_done)
                            v_total = max(0, total - allocated_total)
                        else:
                            v_done = round(done * span_len / max(1, _total_pooled))
                            v_total = round(total * span_len / max(1, _total_pooled))
                        allocated_done += v_done
                        allocated_total += v_total
                        v_res["ragas_completed_steps"] = v_done
                        v_res["ragas_total_steps"] = v_total
                        v_res["phase"] = f"RAGAs {_mode}中 ({done}/{total} · {pct}%)"
                        dirty_jobs[s_jid] = job
                    if dirty_jobs and (done == 0 or done >= total or (now - last_save_ts[0]) >= 0.3):
                        for d_job in dirty_jobs.values():
                            _save(d_job)
                        last_save_ts[0] = now

            judged = _judge(
                pooled_samples,
                scope,
                max_workers=ragas_concurrency,
                progress_cb=_on_batch_ragas_progress,
                selected_metrics=selected_metrics,
            )
            if any(_is_cancelled(jid) for jid in job_ids):
                break
            if len(judged) != len(pooled_samples):
                raise ValueError("RAGAs 返回的评分数量与问题数量不一致。")
            preserve_unselected = bool(selected_metrics)
            for jid, v_name, start, end, v_samples in spans:
                for row, scores in zip(v_samples, judged[start:end]):
                    row_metrics = row.setdefault("metrics", {})
                    if preserve_unselected:
                        if _is_llm_only_sample(row):
                            for r_key in (
                                "llm_context_precision_with_reference",
                                "context_precision",
                                "context_recall",
                                "llm_context_recall",
                                "faithfulness",
                            ):
                                row_metrics.pop(r_key, None)
                        for m_key, m_val in scores.items():
                            if m_val is not None or row_metrics.get(m_key) is None:
                                row_metrics[m_key] = m_val
                    else:
                        for r_key in RAGAS_METRIC_KEYS:
                            row_metrics.pop(r_key, None)
                        row_metrics.update(scores)
                scored_updates[(jid, v_name)] = v_samples

        if any(_is_cancelled(jid) for jid in job_ids):
            for jid in job_ids:
                if not _is_cancelled(jid):
                    cancel_job(jid)
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        with LOCK:
            for jid in job_ids:
                if _is_cancelled(jid):
                    continue
                job = JOBS.get(jid)
                if not job:
                    continue
                for v_name, v_res in (job.get("results") or {}).items():
                    updated_samples = scored_updates.get((jid, v_name))
                    if updated_samples is None:
                        continue
                    summary, valid_counts = _summarize_samples(updated_samples)
                    simple_s, simple_v = _summarize_samples(
                        [row for row in updated_samples if row.get("question_type", "simple") == "simple"]
                    )
                    complex_s, complex_v = _summarize_samples(
                        [row for row in updated_samples if row.get("question_type") == "complex"]
                    )
                    v_res.update(
                        status="completed",
                        phase="完成",
                        ragas_scored=True,
                        samples=updated_samples,
                        metrics=summary,
                        valid_counts=valid_counts,
                        metrics_by_type={"overall": summary, "simple": simple_s, "complex": complex_s},
                        valid_counts_by_type={"overall": valid_counts, "simple": simple_v, "complex": complex_v},
                    )
                    v_res.pop("error", None)
                req = job.get("request")
                if isinstance(req, dict):
                    req["ragas_concurrency"] = ragas_concurrency
                job["status"] = "completed" if all(
                    r.get("status") == "completed" for r in (job.get("results") or {}).values()
                ) else "completed_with_errors"
                job["finished_at"] = now_iso
                job.pop("error", None)
                _save(job)
    except Exception as exc:
        from loguru import logger
        logger.exception("Batch RAGAs evaluation failed")
        err_msg = _error(exc)
        with LOCK:
            for jid in job_ids:
                if _is_cancelled(jid):
                    continue
                job = JOBS.get(jid)
                if not job:
                    continue
                job.update(status="failed", error=err_msg)
                for v_res in (job.get("results") or {}).values():
                    if v_res.get("status") == "running":
                        v_res["status"] = "failed"
                        v_res["phase"] = "RAGAs 评分失败"
                        v_res["error"] = err_msg
                        existing_samples = v_res.get("samples") or []
                        if existing_samples and not v_res.get("metrics"):
                            summary, valid_counts = _summarize_samples(existing_samples)
                            v_res["metrics"] = summary
                            v_res["valid_counts"] = valid_counts
                _save(job)


def start_ragas_evaluation(
    job_ids: list[str],
    ragas_concurrency: int = 16,
    selected_metrics: Optional[list[str]] = None,
) -> dict:
    """Start asynchronous high-concurrency RAGAs scoring across one or more completed jobs."""
    unique_ids = list(dict.fromkeys(job_ids))
    with LOCK:
        if any(job["status"] in ("queued", "running") for job in JOBS.values()):
            raise RuntimeError("已有一批评估正在运行，请等待完成后再启动。")
        loaded_jobs: list[tuple[str, dict]] = []
        for jid in unique_ids:
            job = get_job(jid)
            has_samples = any(
                isinstance(v_res, dict) and bool(v_res.get("samples"))
                for v_res in (job.get("results") or {}).values()
            )
            if not has_samples:
                raise ValueError(f"批次 {jid[:8]} 没有可评分的样本。")
            loaded_jobs.append((jid, copy.deepcopy(job)))

        if selected_metrics:
            phase_label = f"RAGAs 指定指标重评中(并发度={ragas_concurrency})"
        else:
            phase_label = f"RAGAs 批量评分中(并发度={ragas_concurrency})"
        for jid, job in loaded_jobs:
            CANCEL_EVENTS[jid] = threading.Event()
            job["status"] = "running"
            job.pop("error", None)
            for v_res in (job.get("results") or {}).values():
                if isinstance(v_res, dict) and v_res.get("samples"):
                    v_res["status"] = "running"
                    v_res["phase"] = phase_label
                    v_res["ragas_completed_steps"] = 0
                    v_res["ragas_total_steps"] = 0
                    v_res.pop("error", None)
            JOBS[jid] = job
            _save(job)

        threading.Thread(
            target=_run_ragas_for_jobs,
            args=(unique_ids, ragas_concurrency, selected_metrics),
            daemon=True,
            name="ragas-batch-eval",
        ).start()
        return {
            "job_ids": unique_ids,
            "ragas_concurrency": ragas_concurrency,
            "selected_metrics": selected_metrics,
            "jobs": [copy.deepcopy(JOBS[jid]) for jid in unique_ids],
        }


def cancel_job(job_id: str) -> dict:
    """Stop a queued or running evaluation job immediately and preserve partial samples."""
    if not job_id.isalnum() or len(job_id) != 32:
        raise FileNotFoundError(job_id)
    CANCEL_EVENTS.setdefault(job_id, threading.Event()).set()
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            path = RUN_DIR / f"{job_id}.json"
            if not path.is_file():
                raise FileNotFoundError(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
            JOBS[job_id] = job
        if job.get("status") in ("queued", "running"):
            job["status"] = "cancelled"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            for v_res in job.get("results", {}).values():
                if v_res.get("status") in ("queued", "running"):
                    v_res["status"] = "cancelled"
                    v_res["phase"] = "已手动停止"
                    partial = [s for s in (v_res.get("samples") or []) if isinstance(s, dict)]
                    if partial and not v_res.get("metrics"):
                        summary, valid_counts = _summarize_samples(partial)
                        simple_p = [row for row in partial if row.get("question_type", "simple") == "simple"]
                        complex_p = [row for row in partial if row.get("question_type") == "complex"]
                        simple_s, simple_v = _summarize_samples(simple_p)
                        complex_s, complex_v = _summarize_samples(complex_p)
                        v_res["metrics"] = summary
                        v_res["valid_counts"] = valid_counts
                        v_res["metrics_by_type"] = {"overall": summary, "simple": simple_s, "complex": complex_s}
                        v_res["valid_counts_by_type"] = {"overall": valid_counts, "simple": simple_v, "complex": complex_v}
            _save(job)
        return copy.deepcopy(job)


def get_last_request_prefs() -> Optional[dict]:
    """Return variant selection, scope, and count preferences from the most recent evaluation job."""
    latest_job = None
    with LOCK:
        if JOBS:
            latest_job = max(JOBS.values(), key=lambda j: j.get("created_at", ""))
        elif RUN_DIR.exists():
            paths = sorted(RUN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in paths:
                try:
                    latest_job = json.loads(path.read_text(encoding="utf-8"))
                    break
                except (OSError, ValueError):
                    continue
    if not latest_job:
        return None
    req = latest_job.get("request") or {}
    variants = req.get("variants") or []
    variant_names = [v.get("name") for v in variants if isinstance(v, dict) and v.get("name")]
    dataset = req.get("dataset") or []
    first_cfg = (variants[0].get("config") or {}) if variants and isinstance(variants[0], dict) else {}
    return {
        "variant_names": variant_names,
        "scope": req.get("scope", "retrieval"),
        "use_ragas": bool(req.get("use_ragas", True)),
        "concurrency": int(req.get("concurrency", 2) or 2),
        "ragas_concurrency": int(req.get("ragas_concurrency") or 16),
        "top_k": int(first_cfg.get("top_k", 5) or 5),
        "candidate_k": int(first_cfg.get("candidate_k", 20) or 20),
        "simple_count": sum(1 for q in dataset if isinstance(q, dict) and q.get("question_type", "simple") == "simple"),
        "complex_count": sum(1 for q in dataset if isinstance(q, dict) and q.get("question_type") == "complex"),
        "total_count": len(dataset),
    }


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
