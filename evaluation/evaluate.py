"""
evaluation/evaluate.py
──────────────────────
Runs RAGAs evaluation on your RAG pipeline and saves scores to
evaluation/latest_scores.json (served by GET /metrics).

Usage:
    python -m evaluation.evaluate

What it measures
────────────────
  faithfulness      — does the answer stick to the retrieved context?
  answer_relevancy  — is the answer relevant to the question?
  context_precision — are the retrieved chunks actually useful?
  context_recall    — are all necessary facts retrieved?

Add your own test questions to EVAL_DATASET below.
"""

from __future__ import annotations
import json
import math
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from loguru import logger
from ragas import evaluate
from ragas.run_config import RunConfig
from evaluation.judge import DeepSeekJudge
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.agent import ask
from app.llm import create_llm
from embeddings.embedder import EMBED_MODEL
from evaluation.eval_data import EVAL_DATASET

load_dotenv()

SCORES_FILE = Path("evaluation/latest_scores.json")
SCORES_FILE.parent.mkdir(exist_ok=True)

# ── Use DeepSeek as the RAGAs judge LLM ─────────────────────────────────
RAGAS_LLM = DeepSeekJudge(create_llm(temperature=0))

# Apply the LLM to metrics that need a judge
for _metric in (faithfulness, answer_relevancy, context_precision, context_recall):
    _metric.llm = RAGAS_LLM  # type: ignore[attr-defined]

# ── Replace/extend with questions about YOUR actual documents ──────────────
# ground_truth: the ideal answer (used for context_recall only)
# EVAL_DATASET = [
#     {
#         "question":     "What is the main topic of the document?",
#         "ground_truth": "The document covers the primary subject matter in detail.",
#     },
#     {
#         "question":     "What are the key findings or conclusions?",
#         "ground_truth": "The key findings are summarised in the conclusion section.",
#     },
#     {
#         "question":     "What methodology was used?",
#         "ground_truth": "The methodology section describes the approach taken.",
#     },
#     {
#         "question":     "Who are the main authors or contributors?",
#         "ground_truth": "The authors are listed on the title page.",
#     },
#     {
#         "question":     "What recommendations are made?",
#         "ground_truth": "Recommendations are provided in the final section.",
#     },
# ]


from concurrent.futures import ThreadPoolExecutor, as_completed

def run_evaluation(concurrency: int = 4) -> dict:
    logger.info(f"Running RAGAs evaluation on {len(EVAL_DATASET)} questions (concurrency={concurrency}) …")

    ordered_results = [None] * len(EVAL_DATASET)

    def _eval_one(idx_item):
        idx, item = idx_item
        q = item["question"]
        gt = item["ground_truth"]
        q_type = item.get("question_type", "simple")
        t0 = time.perf_counter()
        result = ask(q)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        answer = result["answer"]
        ctx_texts = [c["text"] for c in result["sources"]]
        trace = result.get("retrieval_trace", [])
        retrieval_count = len(trace) if trace else (result.get("rewrite_count", 0) + 1)
        total_tokens = int(result.get("total_tokens", 0) or 0)
        return idx, {
            "question": q,
            "question_type": q_type,
            "answer": answer,
            "contexts": ctx_texts if ctx_texts else ["No context retrieved."],
            "ground_truth": gt,
            "retrieval_count": retrieval_count,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
        }

    max_workers = min(concurrency, len(EVAL_DATASET)) if EVAL_DATASET else 1
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eval-runner") as executor:
        futures = [executor.submit(_eval_one, (i, item)) for i, item in enumerate(EVAL_DATASET)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            ordered_results[idx] = res

    questions = [r["question"] for r in ordered_results]
    answers = [r["answer"] for r in ordered_results]
    contexts = [r["contexts"] for r in ordered_results]
    ground_truths = [r["ground_truth"] for r in ordered_results]

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # Explicit local embeddings prevent RAGAs from defaulting to OpenAI embeddings.
    eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    ))
    result = evaluate(
        dataset,
        llm=RAGAS_LLM,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=max(1, int(concurrency)), timeout=120, max_retries=2),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # RAGAs 0.2 returns EvaluationResult; aggregate per-question metric scores (overall, simple, complex).
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    frame = result.to_pandas()
    rows = []
    for idx, res_item in enumerate(ordered_results):
        row_data = {
            "question_type": res_item["question_type"],
            "retrieval_count": res_item["retrieval_count"],
            "total_tokens": res_item["total_tokens"],
            "latency_ms": res_item["latency_ms"],
        }
        for m_name in metric_names:
            val = frame.iloc[idx][m_name]
            row_data[m_name] = float(val) if val is not None and math.isfinite(float(val)) else None
        rows.append(row_data)

    def _group_scores(subset: list[dict]) -> dict:
        if not subset:
            return {}
        out = {"count": len(subset)}
        for m_name in metric_names:
            vals = [r[m_name] for r in subset if r.get(m_name) is not None]
            out[m_name] = round(sum(vals) / len(vals), 4) if vals else 0.0
        out["avg_retrieval_count"] = round(sum(r["retrieval_count"] for r in subset) / len(subset), 2)
        out["avg_tokens"] = round(sum(r["total_tokens"] for r in subset) / len(subset), 1)
        out["latency_ms"] = round(sum(r["latency_ms"] for r in subset) / len(subset), 2)
        return out

    scores = _group_scores(rows)
    simple_rows = [r for r in rows if r["question_type"] == "simple"]
    complex_rows = [r for r in rows if r["question_type"] == "complex"]
    if simple_rows:
        scores["simple"] = _group_scores(simple_rows)
    if complex_rows:
        scores["complex"] = _group_scores(complex_rows)

    SCORES_FILE.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Scores saved to {SCORES_FILE}")

    def _print_group(title: str, grp: dict) -> None:
        if not grp:
            return
        print(f"\n── {title} ({grp.get('count', 0)} 题) ─────────────────────")
        for m_name in metric_names:
            val = grp.get(m_name, 0.0)
            bar = "█" * int(val * 20)
            print(f"  {m_name:<25} {val:.4f}  {bar}")
        print(f"  {'avg_retrieval_count':<25} {grp.get('avg_retrieval_count', 0):.2f} 次 (平均检索轮次)")
        print(f"  {'avg_tokens':<25} {grp.get('avg_tokens', 0):.1f} tokens (平均消耗)")
        print(f"  {'latency_ms':<25} {grp.get('latency_ms', 0):.2f} ms (平均耗时)")

    _print_group("RAGAs Overall Results", scores)
    if "simple" in scores:
        _print_group("Simple Questions", scores["simple"])
    if "complex" in scores:
        _print_group("Complex Questions", scores["complex"])
    print("─────────────────────────────────────────────────\n")

    return scores


if __name__ == "__main__":
    run_evaluation()
